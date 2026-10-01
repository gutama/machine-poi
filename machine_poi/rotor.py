"""Opt-in G1: real Euclidean Cl(r,0), implemented with low-rank vectors.

For orthonormal u,t, B=u wedge t, R=exp(-theta B/2), and R z reverse(R)
rotates u toward t. No dense multivectors, learned metric, or live adaptation.
"""

import math
from collections import Counter
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RotorArtifact:
    basis: torch.Tensor  # [hidden, rank], orthonormal columns
    target: torch.Tensor  # fixed training-derived coordinates [rank]
    metadata: dict
    tolerance: float = 1e-6

    def __post_init__(self):
        q, s = self.basis.detach().double().cpu().clone(), self.target.detach().double().cpu().clone()
        if q.ndim != 2 or not 2 <= q.shape[1] <= q.shape[0] or s.shape != (q.shape[1],):
            raise ValueError("Rotor needs hidden x rank basis, rank >= 2, and rank-vector target")
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance <= 1e-3:
            raise ValueError("Invalid rotor tolerance")
        if not torch.isfinite(q).all() or not torch.isfinite(s).all():
            raise ValueError("Non-finite rotor artifact")
        if not torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=q.dtype), atol=self.tolerance, rtol=0):
            raise ValueError("Rotor basis must be orthonormal")
        object.__setattr__(self, "basis", q)
        object.__setattr__(self, "target", s)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def copy(self):
        return RotorArtifact(self.basis, self.target, self.metadata, self.tolerance)


def fit_rotor(samples, target, rank, metadata, tolerance=1e-6):
    """Target-first basis plus training SVD; preserve the entire target.

    All rows must be training activations from the named inference checkpoint.
    Reject insufficient numerical rank instead of inventing completion vectors.
    """
    x, d = samples.detach().double().cpu(), target.detach().double().cpu()
    if x.ndim != 2 or d.shape != (x.shape[1],) or type(rank) is not int:
        raise ValueError("Invalid training dimensions/rank")
    if not 2 <= rank <= min(x.shape) or metadata.get("split") != "train":
        raise ValueError("Rotor fit needs training-only samples and feasible rank >= 2")
    if not torch.isfinite(x).all() or not torch.isfinite(d).all() or d.norm() <= tolerance:
        raise ValueError("Training target must be finite and meaningful")
    first = d / d.norm()
    residual = x - (x @ first).unsqueeze(-1) * first
    _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    if int((singular > tolerance * max(float(singular[0]), 1)).sum()) < rank - 1:
        raise ValueError("Insufficient numerical training rank")
    columns = [first]
    for row in vh[:rank - 1]:
        # Reorthogonalize to suppress SVD round-off along the target.
        for col in columns:
            row = row - (row @ col) * col
        row = row / row.norm()
        if row[row.abs().argmax()] < 0:
            row = -row
        columns.append(row)
    q = torch.stack(columns, dim=1)
    return RotorArtifact(q, q.T @ d, {**metadata, "rank": rank, "algebra": f"Cl({rank},0)"}, tolerance)


def rotate_hidden(hidden, artifact, max_angle):
    """Return hidden states and per-token diagnostics, retaining h_perp.

    Zero projected states/targets, parallel/antipodal and near-degenerate
    tangents are exact no-ops. Non-finite inputs or outputs abort the condition.
    Angles are radians, capped by max_angle and the angle to the fixed target.
    """
    if not math.isfinite(max_angle) or not 0 <= max_angle <= math.pi:
        raise ValueError("Rotor angle must be finite radians in [0, pi]")
    if hidden.ndim < 1 or hidden.numel() == 0 or hidden.shape[-1] != artifact.basis.shape[0] or not hidden.is_floating_point():
        raise ValueError("Rotor hidden dimension/dtype mismatch")
    if not torch.isfinite(hidden).all():
        raise ValueError("Non-finite hidden state: experimental condition aborted")
    # Compute at least float32; return the original dtype, and measure its error.
    h = hidden.double() if hidden.dtype == torch.float64 else hidden.float()
    q, s = artifact.basis.to(h.device, h.dtype), artifact.target.to(h.device, h.dtype)
    z = h @ q
    radius = z.norm(dim=-1)
    eps = artifact.tolerance
    u = z / radius.clamp_min(eps).unsqueeze(-1)
    target = s / s.norm().clamp_min(eps)
    cosine = (u @ target).clamp(-1, 1)
    tangent = target - cosine.unsqueeze(-1) * u
    tangent_norm = tangent.norm(dim=-1)
    t = tangent / tangent_norm.clamp_min(eps).unsqueeze(-1)
    angle = torch.acos(cosine).clamp(max=max_angle)
    reasons = torch.zeros_like(radius, dtype=torch.int64)  # 0 rotated
    reasons = torch.where((tangent_norm <= eps) | (cosine.abs() == 1), torch.where(cosine >= 0, 2, 3), reasons)
    reasons = torch.where((radius <= eps) | (s.norm() <= eps), 1, reasons)
    if max_angle == 0:
        reasons = torch.full_like(reasons, 4)
    angle = torch.where(reasons == 0, angle, 0)
    z_new = radius.unsqueeze(-1) * (u * angle.cos().unsqueeze(-1) + t * angle.sin().unsqueeze(-1))
    # h + Q(z_new-z) equals Qz_new + h_perp, with fewer cancellation errors.
    modified = (h + (z_new - z) @ q.T).to(hidden.dtype)
    modified = torch.where((reasons != 0).unsqueeze(-1), hidden, modified)
    if not torch.isfinite(modified).all():
        raise ValueError("Non-finite rotor output: experimental condition aborted")
    original_norm = h.norm(dim=-1)
    relative = (modified.to(h.dtype) - h).norm(dim=-1) / original_norm.clamp_min(eps)
    norm_error = (modified.to(h.dtype).norm(dim=-1) - original_norm).abs() / original_norm.clamp_min(eps)
    return modified, {"relative_displacement": relative, "relative_norm_error": norm_error,
                      "angle_rad": angle, "reason": reasons}


class RotorStats:
    """Bounded scalar summaries across prefill/decode; no retained activations."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens = 0
        self.displacement = self.angle = self.norm_error = 0.0
        self.max_norm_error = 0.0
        self.noops = Counter()

    def update(self, diagnostics):
        names = {1: "zero", 2: "parallel_or_near_parallel", 3: "antipodal_or_near_antipodal", 4: "zero_angle"}
        self.tokens += diagnostics["reason"].numel()
        self.displacement += float(diagnostics["relative_displacement"].sum())
        self.angle += float(diagnostics["angle_rad"].sum())
        errors = diagnostics["relative_norm_error"]
        self.norm_error += float(errors.sum())
        self.max_norm_error = max(self.max_norm_error, float(errors.max()))
        for key, name in names.items():
            self.noops[name] += int((diagnostics["reason"] == key).sum())

    def summary(self):
        n = max(self.tokens, 1)
        return {"tokens": self.tokens, "mean_relative_displacement": self.displacement / n,
                "mean_angle_rad": self.angle / n, "mean_relative_norm_error": self.norm_error / n,
                "max_relative_norm_error": self.max_norm_error, "no_op_reasons": dict(self.noops)}


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().double().cpu().numpy().tobytes()).hexdigest()


def save_rotors(path, artifacts, metadata):
    """Numeric NPZ only, atomic save, with basis/target hashes and provenance."""
    import json
    import os
    import tempfile
    from pathlib import Path
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records, arrays = {}, {}
    for layer, artifact in artifacts.items():
        records[str(layer)] = {**artifact.metadata, "basis_sha256": tensor_hash(artifact.basis),
                               "target_sha256": tensor_hash(artifact.target)}
        arrays[f"basis_{layer}"] = artifact.basis.numpy()
        arrays[f"target_{layer}"] = artifact.target.numpy()
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            name = handle.name
            np.savez(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                     records=np.array(json.dumps(records, sort_keys=True)), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_rotors(path, metadata):
    import json
    import zipfile
    from pathlib import Path
    import numpy as np
    from .steering_cache import CacheMismatchError
    if Path(path).stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Rotor cache exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Expanded rotor cache exceeds size limit")
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["metadata"].item())) != metadata:
            raise CacheMismatchError("Stale rotor checkpoint/training/configuration")
        records = json.loads(str(data["records"].item()))
        if set(records) != {str(i) for i in metadata["layers"]}:
            raise ValueError("Rotor layer mismatch")
        keys = {"metadata", "records"} | {f"{name}_{layer}" for layer in records for name in ("basis", "target")}
        if set(data.files) != keys:
            raise ValueError("Unknown rotor cache fields")
        artifacts = {}
        for layer, record in records.items():
            if record.get("split") != "train" or any(record.get(k) != v for k, v in metadata.items()):
                raise ValueError("Rotor was not fitted on training data")
            q, s = data[f"basis_{layer}"], data[f"target_{layer}"]
            if q.dtype.kind != "f" or s.dtype.kind != "f":
                raise ValueError("Rotor cache must contain floating arrays")
            artifact = RotorArtifact(torch.from_numpy(q.copy()), torch.from_numpy(s.copy()),
                                     record, metadata["rotor_tolerance"])
            if (tensor_hash(artifact.basis) != record["basis_sha256"] or
                    tensor_hash(artifact.target) != record["target_sha256"] or
                    artifact.basis.shape != (metadata["hidden_size"], metadata["rotor_rank"])):
                raise ValueError("Rotor artifact hashes/dimensions mismatch")
            artifacts[int(layer)] = artifact
        return artifacts
