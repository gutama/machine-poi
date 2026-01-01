"""
Steering Vector Extraction Module

Converts text embeddings into steering vectors that can be injected
into LLM hidden states during inference.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


class SteeringVectorExtractor:
    """
    Extracts and projects steering vectors from text embeddings
    to match LLM hidden state dimensions.

    Implements several projection strategies:
    1. Linear projection: Simple learned linear transformation
    2. Random projection: Fast approximate projection
    3. Contrastive: Creates vectors from positive/negative pairs
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        projection_type: str = "linear",
        device: Optional[str] = None,
    ):
        """
        Initialize the steering vector extractor.

        Args:
            source_dim: Dimension of source embeddings
            target_dim: Dimension of target LLM hidden states
            projection_type: Type of projection (linear, random, contrastive)
            device: Device for computation
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.projection_type = projection_type

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.projection_matrix = None
        self._initialize_projection()

    def _initialize_projection(self):
        """Initialize the projection matrix based on type."""
        if self.projection_type == "random":
            # Random Gaussian projection (preserves distances approximately)
            proj = np.random.randn(self.source_dim, self.target_dim)
            proj = proj / np.sqrt(self.source_dim)  # Normalize
            self.projection_matrix = torch.tensor(
                proj, dtype=torch.float32, device=self.device
            )
        elif self.projection_type == "linear":
            # Learnable linear projection (initialized with Xavier)
            self.projection_matrix = torch.nn.Parameter(
                torch.empty(self.source_dim, self.target_dim, device=self.device)
            )
            torch.nn.init.xavier_uniform_(self.projection_matrix)
        elif self.projection_type == "orthogonal":
            # Orthogonal projection using SVD
            proj = np.random.randn(self.source_dim, self.target_dim)
            u, s, vh = np.linalg.svd(proj, full_matrices=False)
            proj = u @ vh
            self.projection_matrix = torch.tensor(
                proj, dtype=torch.float32, device=self.device
            )

    def project_embedding(
        self,
        embedding: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Project a text embedding to LLM hidden state space.

        Args:
            embedding: Source embedding [source_dim] or [batch, source_dim]
            normalize: Whether to L2 normalize the result

        Returns:
            Projected vector [target_dim] or [batch, target_dim]
        """
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32, device=self.device)
        else:
            embedding = embedding.to(self.device)

        # Handle batched and unbatched inputs
        squeeze = False
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            squeeze = True

        # Project
        projected = embedding @ self.projection_matrix

        if normalize:
            projected = torch.nn.functional.normalize(projected, dim=-1)

        if squeeze:
            projected = projected.squeeze(0)

        return projected

    def create_steering_vector(
        self,
        positive_embedding: np.ndarray,
        negative_embedding: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Create a steering vector from embeddings.

        For contrastive steering: vector = positive - negative
        For single-direction steering: vector = positive

        Args:
            positive_embedding: Embedding representing desired direction
            negative_embedding: Optional embedding for contrast
            scale: Scaling factor for the vector

        Returns:
            Steering vector [target_dim]
        """
        pos_proj = self.project_embedding(positive_embedding, normalize=False)

        if negative_embedding is not None:
            neg_proj = self.project_embedding(negative_embedding, normalize=False)
            steering_vector = pos_proj - neg_proj
        else:
            steering_vector = pos_proj

        # Normalize and scale
        steering_vector = torch.nn.functional.normalize(steering_vector, dim=-1)
        steering_vector = steering_vector * scale

        return steering_vector

    def create_multi_layer_vectors(
        self,
        embedding: np.ndarray,
        n_layers: int,
        scale_per_layer: Optional[List[float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Create steering vectors for multiple layers.

        Different layers may need different scaling or projections.

        Args:
            embedding: Source embedding
            n_layers: Number of layers in target LLM
            scale_per_layer: Optional per-layer scaling factors

        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        if scale_per_layer is None:
            # Default: focus on middle layers (empirically most effective)
            scale_per_layer = []
            for i in range(n_layers):
                # Bell curve centered on middle layers
                center = n_layers / 2
                scale = np.exp(-0.5 * ((i - center) / (n_layers / 4)) ** 2)
                scale_per_layer.append(scale)

        vectors = {}
        base_vector = self.project_embedding(embedding, normalize=True)

        for i in range(n_layers):
            vectors[i] = base_vector * scale_per_layer[i]

        return vectors

    def save(self, path: Union[str, Path]):
        """Save the projection matrix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "projection_matrix": self.projection_matrix.cpu().numpy()
            if isinstance(self.projection_matrix, torch.Tensor)
            else self.projection_matrix.detach().cpu().numpy(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "projection_type": self.projection_type,
        }
        np.savez(path, **state)

    def load(self, path: Union[str, Path]):
        """Load a saved projection matrix."""
        data = np.load(path)
        self.projection_matrix = torch.tensor(
            data["projection_matrix"],
            dtype=torch.float32,
            device=self.device,
        )
        self.source_dim = int(data["source_dim"])
        self.target_dim = int(data["target_dim"])
        self.projection_type = str(data["projection_type"])


class ContrastiveSteeringExtractor(SteeringVectorExtractor):
    """
    Creates steering vectors using contrastive activation addition (CAA).

    This approach computes vectors by averaging activation differences
    between positive and negative example pairs.
    """

    def __init__(
        self,
        target_dim: int,
        device: Optional[str] = None,
    ):
        """
        Initialize contrastive extractor.

        Note: source_dim is not needed as we work directly with activations.
        """
        self.target_dim = target_dim
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.steering_vectors = {}

    def compute_from_activations(
        self,
        positive_activations: List[torch.Tensor],
        negative_activations: List[torch.Tensor],
        layer_idx: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute steering vector from paired activations.

        Args:
            positive_activations: List of activations for positive examples
            negative_activations: List of activations for negative examples
            layer_idx: Which layer these activations are from
            normalize: Whether to normalize the result

        Returns:
            Steering vector for this layer
        """
        # Average the differences
        diffs = []
        for pos, neg in zip(positive_activations, negative_activations):
            # Use the last token position or mean across positions
            if pos.dim() == 2:  # [seq_len, hidden_dim]
                pos = pos.mean(dim=0)
                neg = neg.mean(dim=0)
            diffs.append(pos - neg)

        steering_vector = torch.stack(diffs).mean(dim=0)

        if normalize:
            steering_vector = torch.nn.functional.normalize(steering_vector, dim=-1)

        self.steering_vectors[layer_idx] = steering_vector
        return steering_vector

    def get_vector(self, layer_idx: int, scale: float = 1.0) -> Optional[torch.Tensor]:
        """Get the steering vector for a layer."""
        if layer_idx in self.steering_vectors:
            return self.steering_vectors[layer_idx] * scale
        return None
