"""Tests for workspace-inspired steering diagnostics."""

from types import SimpleNamespace

import pytest
import torch


def test_summarize_layer_steering_reports_expected_metrics():
    from src.workspace_diagnostics import summarize_layer_steering

    activation = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    steering = torch.tensor([1.0, 0.0])

    diagnostics = summarize_layer_steering(activation, steering, coefficient=0.5)

    assert diagnostics.activation_norm == pytest.approx(1.5)
    assert diagnostics.steering_norm == pytest.approx(1.0)
    assert diagnostics.mean_cosine_similarity == pytest.approx(0.5)
    assert diagnostics.mean_projection_magnitude == pytest.approx(0.5)
    assert diagnostics.relative_perturbation == pytest.approx(1 / 3)


def test_summarize_layer_steering_validates_hidden_dimension():
    from src.workspace_diagnostics import summarize_layer_steering

    with pytest.raises(ValueError):
        summarize_layer_steering(torch.zeros(1, 2, 3), torch.zeros(2))


def test_summarize_layer_steering_reports_raw_zero_norms():
    from src.workspace_diagnostics import summarize_layer_steering

    diagnostics = summarize_layer_steering(
        activation=torch.zeros(1, 2, 3),
        steering_vector=torch.zeros(3),
    )

    assert diagnostics.activation_norm == pytest.approx(0.0)
    assert diagnostics.steering_norm == pytest.approx(0.0)
    assert diagnostics.mean_cosine_similarity == pytest.approx(0.0)
    assert diagnostics.mean_projection_magnitude == pytest.approx(0.0)
    assert diagnostics.relative_perturbation == pytest.approx(0.0)


def _causal_uniform_attention(seq_len: int) -> torch.Tensor:
    """Row-normalized lower-triangular attention weights."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask / mask.sum(dim=-1, keepdim=True)


def test_connection_bivectors_are_antisymmetric():
    from src.workspace_diagnostics import connection_bivectors

    torch.manual_seed(0)
    seq_len, dim = 5, 4
    omegas = connection_bivectors(
        _causal_uniform_attention(seq_len),
        torch.randn(seq_len, dim),
        torch.randn(seq_len, dim),
    )

    assert omegas.shape == (seq_len, dim, dim)
    assert torch.allclose(omegas, -omegas.transpose(-1, -2), atol=1e-6)


def test_connection_bivectors_validates_shapes():
    from src.workspace_diagnostics import connection_bivectors

    with pytest.raises(ValueError):
        connection_bivectors(torch.ones(3, 2), torch.zeros(3, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError):
        connection_bivectors(torch.ones(3, 3), torch.zeros(2, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError):
        connection_bivectors(torch.ones(3, 3), torch.zeros(3, 4), torch.zeros(3, 5))


def test_attention_transport_coplanar_generators_commute():
    from src.workspace_diagnostics import summarize_attention_transport

    # All queries along e1 and all value deviations along e2 keep every ω_t
    # in the e1∧e2 plane, so successive generators commute exactly: ρ = 0.
    seq_len, dim = 5, 4
    queries = torch.zeros(seq_len, dim)
    queries[:, 0] = 1.0
    values = torch.zeros(seq_len, dim)
    values[:, 1] = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0])

    diagnostics = summarize_attention_transport(
        _causal_uniform_attention(seq_len), queries, values
    )

    assert diagnostics.commutator_energy == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.non_abelian_ratio == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.variation_energy > 0.0
    assert diagnostics.num_segments == seq_len - 1
    # Same-plane rotations still compose to a net rotation around loops.
    assert diagnostics.num_loops > 0
    assert diagnostics.mean_holonomy > 0.0


def test_attention_transport_noncoplanar_generators_do_not_commute():
    from src.workspace_diagnostics import summarize_attention_transport

    torch.manual_seed(1)
    seq_len, dim = 6, 4
    diagnostics = summarize_attention_transport(
        _causal_uniform_attention(seq_len),
        torch.randn(seq_len, dim),
        torch.randn(seq_len, dim),
    )

    assert diagnostics.commutator_energy > 0.0
    assert diagnostics.non_abelian_ratio > 0.0
    assert diagnostics.non_abelian_ratio < 1.0
    assert diagnostics.mean_holonomy > 0.0
    assert diagnostics.max_holonomy >= diagnostics.mean_holonomy


def test_attention_transport_identical_values_is_flat():
    from src.workspace_diagnostics import summarize_attention_transport

    # Identical values give zero deviations, hence ω_t = 0 everywhere:
    # flat connection, no curvature, identity transport, zero holonomy.
    seq_len, dim = 4, 4
    diagnostics = summarize_attention_transport(
        _causal_uniform_attention(seq_len),
        torch.ones(seq_len, dim),
        torch.ones(seq_len, dim),
    )

    assert diagnostics.variation_energy == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.commutator_energy == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.mean_curvature_norm == pytest.approx(0.0, abs=1e-6)
    assert diagnostics.mean_holonomy == pytest.approx(0.0, abs=1e-6)


def test_attention_transport_short_sequence_is_degenerate():
    from src.workspace_diagnostics import summarize_attention_transport

    diagnostics = summarize_attention_transport(
        torch.ones(1, 1), torch.ones(1, 4), torch.ones(1, 4)
    )

    assert diagnostics.num_segments == 0
    assert diagnostics.num_loops == 0
    assert diagnostics.non_abelian_ratio == pytest.approx(0.0)


def test_summarize_attention_transport_heads_and_pooling():
    from src.workspace_diagnostics import (
        pooled_non_abelian_ratio,
        summarize_attention_transport_heads,
    )

    torch.manual_seed(2)
    num_heads, seq_len, dim = 3, 5, 4
    attn = _causal_uniform_attention(seq_len).expand(num_heads, -1, -1)
    diagnostics = summarize_attention_transport_heads(
        attn,
        torch.randn(num_heads, seq_len, dim),
        torch.randn(num_heads, seq_len, dim),
    )

    assert sorted(diagnostics) == [0, 1, 2]
    ratios = [d.non_abelian_ratio for d in diagnostics.values()]
    pooled = pooled_non_abelian_ratio(diagnostics.values())
    assert min(ratios) <= pooled <= max(ratios)


def test_summarize_attention_transport_heads_validates_shapes():
    from src.workspace_diagnostics import summarize_attention_transport_heads

    with pytest.raises(ValueError):
        summarize_attention_transport_heads(
            torch.ones(3, 3), torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)
        )
    with pytest.raises(ValueError):
        summarize_attention_transport_heads(
            torch.ones(2, 3, 3), torch.zeros(1, 3, 4), torch.zeros(2, 3, 4)
        )


def test_summarize_steering_hooks_skips_incomplete_hooks():
    from src.workspace_diagnostics import summarize_steering_hooks

    hooks = {
        0: SimpleNamespace(
            captured_activation=torch.ones(1, 1, 2),
            steering_vector=torch.tensor([1.0, 0.0]),
            coefficient=1.0,
        ),
        1: SimpleNamespace(captured_activation=None, steering_vector=torch.ones(2)),
    }

    diagnostics = summarize_steering_hooks(hooks)

    assert list(diagnostics) == [0]
