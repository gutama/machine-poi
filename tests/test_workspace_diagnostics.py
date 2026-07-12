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
