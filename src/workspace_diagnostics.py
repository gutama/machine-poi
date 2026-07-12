"""
Workspace-inspired diagnostics for activation steering.

These utilities provide lightweight tensor-only measurements that help inspect
whether a steering vector acts like a targeted intermediate-layer intervention
or a broad activation perturbation.
"""

from dataclasses import dataclass
from typing import Dict, Optional

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
