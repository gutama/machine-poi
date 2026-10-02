"""Executed G1 invariants against independent dense Cl(3,0) fixtures."""

import math

import pytest
import torch

from experiments.geometry_sanity import rotor_step
from machine_poi.llm_wrapper import ActivationHook, SteeredLLM
from machine_poi.rotor import RotorArtifact, fit_rotor, rotate_hidden, load_rotors, save_rotors
from machine_poi.steering_cache import CacheMismatchError


def artifact(hidden=7, rank=3):
    generator = torch.Generator().manual_seed(42)
    q, _ = torch.linalg.qr(torch.randn(hidden, rank, generator=generator, dtype=torch.double))
    return RotorArtifact(q, torch.arange(1, rank + 1, dtype=torch.double), {"split": "train"})


def test_random_sandwich_sign_norm_residual_and_displacement():
    a = artifact()
    rng = torch.Generator().manual_seed(72)
    for _ in range(50):
        h = torch.randn(7, generator=rng, dtype=torch.double)
        changed, diag = rotate_hidden(h, a, .2)
        z = h @ a.basis
        ga, _, reason, theta, _ = rotor_step(z.tolist(), a.target.tolist(), .2)
        assert reason == "rotated"
        assert torch.allclose(changed @ a.basis, torch.tensor(ga, dtype=torch.double), atol=1e-10)
        assert torch.allclose(changed.norm(), h.norm(), atol=1e-10)
        assert torch.allclose(changed - (changed @ a.basis) @ a.basis.T,
                              h - z @ a.basis.T, atol=1e-10)
        expected = float(z.norm() / h.norm()) * 2 * math.sin(theta / 2)
        assert float(diag["relative_displacement"]) == pytest.approx(expected, abs=1e-10)
        assert float(diag["relative_norm_error"]) < 1e-10
    # Positive e1 wedge e2 convention moves e1 toward e2.
    a = RotorArtifact(torch.eye(3, dtype=torch.double), torch.tensor([0., 1., 0.]), {})
    changed, _ = rotate_hidden(torch.tensor([1., 0., 0.], dtype=torch.double), a, .1)
    assert changed.tolist() == pytest.approx([math.cos(.1), math.sin(.1), 0])


@pytest.mark.parametrize("h,target,reason", [([0., 0.], [1., 0.], 1), ([1., 0.], [0., 0.], 1),
    ([1., 0.], [1., 0.], 2), ([1., 0.], [-1., 0.], 3),
    ([1., 0.], [1., 1e-9], 2), ([1., 0.], [-1., 1e-9], 3)])
def test_exact_noops(h, target, reason):
    a = RotorArtifact(torch.eye(2), torch.tensor(target), {})
    h = torch.tensor(h)
    changed, diag = rotate_hidden(h, a, .2)
    assert torch.equal(changed, h)
    assert int(diag["reason"]) == reason


def test_zero_angle_cap_and_low_precision_diagnostics():
    a = artifact()
    h = torch.randn(2, 3, 7)
    changed, diag = rotate_hidden(h, a, 0)
    assert torch.equal(changed, h)
    assert (diag["reason"] == 4).all()
    a = RotorArtifact(torch.eye(2, dtype=torch.double), torch.tensor([math.cos(.01), math.sin(.01)]), {})
    changed, diag = rotate_hidden(torch.tensor([1., 0.]), a, .2)
    assert float(diag["angle_rad"]) == pytest.approx(.01, abs=2e-6)
    assert torch.allclose(changed, a.target.float(), atol=2e-6)
    h = torch.randn(2, 3, 7).to(torch.bfloat16)
    changed, diag = rotate_hidden(h, artifact(), .1)
    assert changed.dtype == h.dtype
    measured = (changed.float().norm(dim=-1) - h.float().norm(dim=-1)).abs() / h.float().norm(dim=-1)
    assert torch.allclose(measured, diag["relative_norm_error"])


@pytest.mark.parametrize("angle", [-1, math.pi + .01, math.nan, math.inf])
def test_invalid_angle(angle):
    with pytest.raises(ValueError):
        rotate_hidden(torch.ones(7), artifact(), angle)


def test_invalid_artifacts_and_nonfinite_abort():
    with pytest.raises(ValueError, match="orthonormal"):
        RotorArtifact(torch.ones(3, 2), torch.ones(2), {})
    with pytest.raises(ValueError):
        RotorArtifact(torch.eye(2), torch.ones(3), {})
    with pytest.raises(ValueError):
        RotorArtifact(torch.ones(3, 1), torch.ones(1), {})
    for bad in (math.nan, math.inf):
        h = torch.ones(7)
        h[0] = bad
        with pytest.raises(ValueError, match="aborted"):
            rotate_hidden(h, artifact(), .1)


def test_fit_training_only_and_represents_target():
    x = torch.randn(10, 7, generator=torch.Generator().manual_seed(4))
    d = torch.arange(1., 8.)
    a = fit_rotor(x, d, 4, {"split": "train", "revision": "a" * 40})
    assert torch.allclose(a.basis @ a.target, d.double(), atol=1e-10)
    for samples, target, rank, split in ((x, d, 4, "test"), (x, d * 0, 4, "train"),
                                        (x, d, 11, "train"), (torch.ones(10, 7), torch.ones(7), 3, "train")):
        with pytest.raises(ValueError):
            fit_rotor(samples, target, rank, {"split": split})


def test_stale_rotor_cache_and_array_tampering(tmp_path):
    import numpy as np
    path = tmp_path / "rotor.npz"
    meta = {"layers": [1], "hidden_size": 7, "rotor_rank": 3, "rotor_tolerance": 1e-6,
            "revision": "a" * 40, "training_ids": ["train-a"]}
    save_rotors(path, {1: RotorArtifact(artifact().basis, artifact().target, {"split": "train", **meta})}, meta)
    loaded = load_rotors(path, meta)[1]
    assert torch.equal(loaded.basis, artifact().basis)
    for key, value in (("revision", "b" * 40), ("training_ids", ["test-b"])):
        with pytest.raises(CacheMismatchError):
            load_rotors(path, {**meta, key: value})
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["target_1"] *= 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="hashes"):
        load_rotors(path, meta)


def test_hook_disabled_and_tuple_restoration_after_failure():
    from unittest.mock import Mock
    llm = SteeredLLM(model_name="qwen2.5-0.5b", device="cpu")
    llm.model = Mock()
    layer = torch.nn.Identity()
    llm._get_layer_module = lambda _: layer
    # Property introspection uses a model config.
    llm.model.config.hidden_size = 7
    llm.config = {"hidden_size_attr": "hidden_size"}
    previous = llm.register_steering_hook(0, injection_mode="rotor", experimental_rotor=True,
                                         rotor_artifact=artifact(), rotor_max_angle=.1)
    previous.disable()
    h = torch.ones(1, 2, 7)
    assert previous(layer, (), (h, "rest"))[0] is h
    with pytest.raises(ValueError, match="aborted"):
        with llm.steering_session():
            llm.clear_steering()
            new = llm.register_steering_hook(0, injection_mode="rotor", experimental_rotor=True,
                                            rotor_artifact=artifact(), rotor_max_angle=.2)
            new(layer, (), torch.full((1, 1, 7), math.nan))
    assert len(layer._forward_hooks) == 1
    assert not llm.hooks[0].enabled
    assert llm.hooks[0].rotor_max_angle == .1
    assert torch.equal(llm.hooks[0].rotor_artifact.basis, previous.rotor_artifact.basis)
    with pytest.raises(ValueError, match="experimental"):
        llm.register_steering_hook(0, injection_mode="rotor", rotor_artifact=artifact())
    llm.clear_steering()
    assert not layer._forward_hooks


def test_additive_coefficient_cannot_stand_for_angle():
    with pytest.raises(ValueError, match="separate"):
        ActivationHook(0, coefficient=.1, injection_mode="rotor", rotor_artifact=artifact())


def test_invalid_hidden_dimensions_and_options_do_not_load_a_model(monkeypatch):
    for hidden in (torch.tensor(1.), torch.empty(0, 7), torch.ones(8)):
        with pytest.raises(ValueError):
            rotate_hidden(hidden, artifact(), .1)
    llm = SteeredLLM("qwen2.5-0.5b", device="cpu")
    monkeypatch.setattr(llm, "load_model", lambda: pytest.fail("Loaded invalid condition"))
    with pytest.raises(ValueError):
        llm.register_steering_hook(0, injection_mode="rotor", rotor_artifact=artifact())
    with pytest.raises(ValueError):
        llm.register_steering_hook(0, injection_mode="rotor", experimental_rotor=True)
