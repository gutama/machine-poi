"""
Workspace-inspired diagnostics for activation steering.

These utilities provide lightweight tensor-only measurements that help inspect
whether a steering vector acts like a targeted intermediate-layer intervention
or a broad activation perturbation.

The attention-transport section implements the discrete Cartan curvature
diagnostic from "Is Attention Commutative? Quantifying Contextuality via a
Discrete Cartan Curvature Diagnostic" (2026): per-position connection
bivectors ω_t built from attention weights (Eqs. 3-4), the adjacent-segment
curvature split Ω_t = (dω)_t + (ω∧ω)_t (Eqs. 5-7), the non-abelian ratio
ρ = Σ‖ω∧ω‖ / (Σ‖dω‖ + Σ‖ω∧ω‖) (Eq. 8), and holonomy angles from exact
transport maps T_t = exp(−η ω_t) around triangular loops (Eq. 9). ρ ≈ 0
means local transport generators nearly commute (weakly order-sensitive
context routing); larger ρ and holonomy indicate measurable path dependence.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


@dataclass(frozen=True)
class LayerSteeringDiagnostics:
    """Summary metrics for one steered layer."""

    activation_norm: float
    steering_norm: float
    mean_cosine_similarity: float
    mean_projection_magnitude: float
    relative_perturbation: float


def summarize_layer_steering(
    activation: torch.Tensor,
    steering_vector: torch.Tensor,
    coefficient: float = 1.0,
    eps: float = 1e-8,
) -> LayerSteeringDiagnostics:
    """
    Summarize how strongly an activation aligns with a steering vector.

    Args:
        activation: Hidden states with final dimension equal to hidden size.
        steering_vector: Steering direction for the same hidden size.
        coefficient: Effective steering coefficient applied to the vector.
        eps: Numerical-stability constant.
    """
    if activation.shape[-1] != steering_vector.shape[-1]:
        raise ValueError(
            "Activation hidden dimension must match steering vector dimension: "
            f"{activation.shape[-1]} != {steering_vector.shape[-1]}"
        )

    hidden = activation.detach().float()
    vector = steering_vector.detach().float()
    perturbation = vector * coefficient

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    hidden_norms = flat_hidden.norm(dim=-1)
    vector_norm = vector.norm()
    unit_vector = vector / vector_norm.clamp_min(eps)

    cosine = torch.nn.functional.cosine_similarity(
        flat_hidden,
        unit_vector.unsqueeze(0).expand_as(flat_hidden),
        dim=-1,
        eps=eps,
    )
    projection = torch.matmul(flat_hidden, unit_vector)

    activation_norm = hidden_norms.mean()
    return LayerSteeringDiagnostics(
        activation_norm=float(activation_norm.item()),
        steering_norm=float(vector_norm.item()),
        mean_cosine_similarity=float(cosine.mean().item()),
        mean_projection_magnitude=float(projection.abs().mean().item()),
        relative_perturbation=float((perturbation.norm() / activation_norm.clamp_min(eps)).item()),
    )


@dataclass(frozen=True)
class AttentionTransportDiagnostics:
    """Discrete Cartan curvature summary for one attention head."""

    non_abelian_ratio: float
    variation_energy: float
    commutator_energy: float
    mean_curvature_norm: float
    mean_holonomy: float
    max_holonomy: float
    num_segments: int
    num_loops: int


def _bivector_norms(bivectors: torch.Tensor) -> torch.Tensor:
    """‖B‖ = sqrt(-½ tr(B²)) = ‖B‖_F / √2 for antisymmetric B, batched."""
    frob = bivectors.flatten(-2).norm(dim=-1)
    return frob / (2.0 ** 0.5)


def connection_bivectors(
    attn_weights: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build per-position connection bivectors ω_t for one attention head.

        ω_t = Σ_{s≤t} α_{t→s} · B_{t,s},   B_{t,s} = norm(q̂_t ∧ Δv_{t,s})

    with Δv_{t,s} = v_s − v̄_t the centered value deviation (v̄_t is the mean
    value over positions s ≤ t) and u ∧ w = u wᵀ − w uᵀ the wedge product.

    Args:
        attn_weights: [seq, seq] attention weights; row t holds α_{t→s}.
        queries: [seq, head_dim] per-position query vectors for this head.
        values: [seq, head_dim] per-position value vectors for this head.
        eps: Numerical-stability constant.

    Returns:
        [seq, head_dim, head_dim] antisymmetric matrices (ω_t stack).
    """
    if attn_weights.dim() != 2 or attn_weights.shape[0] != attn_weights.shape[1]:
        raise ValueError(f"attn_weights must be square [seq, seq], got {tuple(attn_weights.shape)}")
    seq_len = attn_weights.shape[0]
    if queries.shape[:1] != (seq_len,) or values.shape[:1] != (seq_len,):
        raise ValueError(
            "queries and values must have one row per position: "
            f"attn seq {seq_len}, queries {tuple(queries.shape)}, values {tuple(values.shape)}"
        )
    if queries.shape[-1] != values.shape[-1]:
        raise ValueError(
            "queries and values must share head dimension: "
            f"{queries.shape[-1]} != {values.shape[-1]}"
        )

    alpha = attn_weights.detach().float()
    q = queries.detach().float()
    v = values.detach().float()
    dim = q.shape[-1]
    device = q.device

    # Cumulative means give v̄_t for every prefix v[: t + 1] in one pass.
    counts = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
    v_means = v.cumsum(dim=0) / counts.unsqueeze(-1)

    omegas = torch.zeros(seq_len, dim, dim, dtype=torch.float32, device=device)
    for t in range(seq_len):
        q_norm = q[t].norm()
        if q_norm < eps:
            continue
        q_hat = q[t] / q_norm
        deltas = v[: t + 1] - v_means[t]
        delta_norms = deltas.norm(dim=-1)
        weights = alpha[t, : t + 1]
        mask = (weights >= eps) & (delta_norms >= eps)
        if not bool(mask.any()):
            continue
        delta_hats = deltas[mask] / delta_norms[mask].unsqueeze(-1)
        # Wedge is bilinear: Σ_s α_s (q̂ ∧ δ̂_s) = q̂ ∧ (Σ_s α_s δ̂_s).
        weighted_sum = (weights[mask].unsqueeze(-1) * delta_hats).sum(dim=0)
        omegas[t] = torch.outer(q_hat, weighted_sum) - torch.outer(weighted_sum, q_hat)
    return omegas


def summarize_attention_transport(
    attn_weights: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    eta: float = 1.0,
    max_loop_positions: int = 8,
    eps: float = 1e-8,
) -> AttentionTransportDiagnostics:
    """
    Discrete Cartan curvature diagnostic for one attention head.

    Splits the curvature over adjacent position segments into a variation
    term (dω)_t = ω_{t+1} − ω_t and a commutator term (ω∧ω)_t = [ω_t, ω_{t+1}],
    reports the non-abelian ratio ρ, and measures holonomy angles of
    T_k T_j T_i around triangular loops with T_t = exp(−η ω_t).

    Args:
        attn_weights: [seq, seq] attention weights; row t holds α_{t→s}.
        queries: [seq, head_dim] per-position query vectors for this head.
        values: [seq, head_dim] per-position value vectors for this head.
        eta: Transport step size η in T_t = exp(−η ω_t).
        max_loop_positions: Cap on positions used for triangular loops
            (evenly spaced across the sequence) to bound the O(n³) loop count.
        eps: Numerical-stability constant.
    """
    omegas = connection_bivectors(attn_weights, queries, values, eps=eps)
    seq_len = omegas.shape[0]

    if seq_len < 2:
        return AttentionTransportDiagnostics(
            non_abelian_ratio=0.0,
            variation_energy=0.0,
            commutator_energy=0.0,
            mean_curvature_norm=0.0,
            mean_holonomy=0.0,
            max_holonomy=0.0,
            num_segments=0,
            num_loops=0,
        )

    # Curvature split over adjacent segments (Eqs. 5-7).
    omega_t = omegas[:-1]
    omega_next = omegas[1:]
    dw = omega_next - omega_t
    ww = omega_t @ omega_next - omega_next @ omega_t
    curvature = dw + ww

    dw_norms = _bivector_norms(dw)
    ww_norms = _bivector_norms(ww)
    variation_energy = float(dw_norms.sum().item())
    commutator_energy = float(ww_norms.sum().item())
    rho = commutator_energy / (variation_energy + commutator_energy + eps)

    # Holonomy of T_k T_j T_i around triangles i < j < k (Eq. 9, §3.4).
    holonomies = []
    if seq_len >= 3:
        n_loop = min(seq_len, max(3, max_loop_positions))
        loop_positions = (
            torch.linspace(0, seq_len - 1, n_loop, device=omegas.device).round().long().unique()
        )
        transports = torch.linalg.matrix_exp(-eta * omegas[loop_positions])
        for c in range(2, len(loop_positions)):
            for b in range(1, c):
                for a in range(b):
                    hol = transports[c] @ transports[b] @ transports[a]
                    angles = torch.linalg.eigvals(hol).angle().abs()
                    holonomies.append(float(angles.max().item()))

    return AttentionTransportDiagnostics(
        non_abelian_ratio=rho,
        variation_energy=variation_energy,
        commutator_energy=commutator_energy,
        mean_curvature_norm=float(_bivector_norms(curvature).mean().item()),
        mean_holonomy=float(sum(holonomies) / len(holonomies)) if holonomies else 0.0,
        max_holonomy=max(holonomies) if holonomies else 0.0,
        num_segments=seq_len - 1,
        num_loops=len(holonomies),
    )


def summarize_attention_transport_heads(
    attn_weights: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    eta: float = 1.0,
    max_loop_positions: int = 8,
    eps: float = 1e-8,
) -> Dict[int, AttentionTransportDiagnostics]:
    """
    Per-head transport diagnostics for stacked head tensors.

    Args:
        attn_weights: [num_heads, seq, seq] attention weights.
        queries: [num_heads, seq, head_dim] query vectors.
        values: [num_heads, seq, head_dim] value vectors.
    """
    if attn_weights.dim() != 3:
        raise ValueError(f"attn_weights must be [heads, seq, seq], got {tuple(attn_weights.shape)}")
    num_heads = attn_weights.shape[0]
    if queries.shape[0] != num_heads or values.shape[0] != num_heads:
        raise ValueError(
            "queries and values must have one stack per head: "
            f"attn heads {num_heads}, queries {tuple(queries.shape)}, values {tuple(values.shape)}"
        )

    return {
        head: summarize_attention_transport(
            attn_weights[head],
            queries[head],
            values[head],
            eta=eta,
            max_loop_positions=max_loop_positions,
            eps=eps,
        )
        for head in range(num_heads)
    }


def pooled_non_abelian_ratio(
    diagnostics: Iterable[AttentionTransportDiagnostics],
    eps: float = 1e-8,
) -> float:
    """Energy-weighted ρ pooled across heads/layers (Eq. 8 on summed energies)."""
    variation = 0.0
    commutator = 0.0
    for diag in diagnostics:
        variation += diag.variation_energy
        commutator += diag.commutator_energy
    return commutator / (variation + commutator + eps)


def summarize_steering_hooks(hooks: Dict[int, object]) -> Dict[int, LayerSteeringDiagnostics]:
    """
    Summarize all hooks that expose captured activations and steering vectors.

    The function accepts generic hook-like objects so tests and downstream tools
    can reuse it without importing the runtime LLM wrapper.
    """
    diagnostics: Dict[int, LayerSteeringDiagnostics] = {}
    for layer_idx, hook in hooks.items():
        activation: Optional[torch.Tensor] = getattr(hook, "captured_activation", None)
        steering_vector: Optional[torch.Tensor] = getattr(hook, "steering_vector", None)
        if activation is None or steering_vector is None:
            continue
        coefficient = float(getattr(hook, "coefficient", 1.0))
        diagnostics[layer_idx] = summarize_layer_steering(
            activation=activation,
            steering_vector=steering_vector,
            coefficient=coefficient,
        )
    return diagnostics
