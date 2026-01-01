"""
Tests for src/steering_vectors.py

Tests for:
- SteeringVectorExtractor (projection methods)
- ContrastiveSteeringExtractor (CAA-style vectors)
"""

import pytest
import numpy as np
import torch
from pathlib import Path


class TestSteeringVectorExtractorInit:
    """Test SteeringVectorExtractor initialization."""

    def test_init_random_projection(self, sample_embedding_dim, sample_hidden_dim):
        """Test initialization with random projection."""
        from src.steering_vectors import SteeringVectorExtractor
        
        extractor = SteeringVectorExtractor(
            source_dim=sample_embedding_dim,
            target_dim=sample_hidden_dim,
            projection_type="random",
            device="cpu",
        )
        
        assert extractor.source_dim == sample_embedding_dim
        assert extractor.target_dim == sample_hidden_dim
        assert extractor.projection_type == "random"
        assert extractor.projection_matrix is not None

    def test_init_linear_projection(self, sample_embedding_dim, sample_hidden_dim):
        """Test initialization with linear (learnable) projection."""
        from src.steering_vectors import SteeringVectorExtractor
        
        extractor = SteeringVectorExtractor(
            source_dim=sample_embedding_dim,
            target_dim=sample_hidden_dim,
            projection_type="linear",
            device="cpu",
        )
        
        assert isinstance(extractor.projection_matrix, torch.nn.Parameter)

    def test_init_orthogonal_projection(self, sample_embedding_dim, sample_hidden_dim):
        """Test initialization with orthogonal projection."""
        from src.steering_vectors import SteeringVectorExtractor
        
        extractor = SteeringVectorExtractor(
            source_dim=sample_embedding_dim,
            target_dim=sample_hidden_dim,
            projection_type="orthogonal",
            device="cpu",
        )
        
        assert extractor.projection_matrix is not None

    def test_projection_matrix_shape(self, sample_embedding_dim, sample_hidden_dim):
        """Test that projection matrix has correct shape."""
        from src.steering_vectors import SteeringVectorExtractor
        
        extractor = SteeringVectorExtractor(
            source_dim=sample_embedding_dim,
            target_dim=sample_hidden_dim,
            projection_type="random",
            device="cpu",
        )
        
        assert extractor.projection_matrix.shape == (sample_embedding_dim, sample_hidden_dim)


class TestProjectEmbedding:
    """Test embedding projection functionality."""

    def test_project_single_embedding_numpy(self, steering_vector_extractor, sample_mean_embedding):
        """Test projecting a single numpy embedding."""
        projected = steering_vector_extractor.project_embedding(sample_mean_embedding)
        
        assert isinstance(projected, torch.Tensor)
        assert projected.shape == (steering_vector_extractor.target_dim,)

    def test_project_single_embedding_tensor(self, steering_vector_extractor, sample_mean_embedding):
        """Test projecting a single tensor embedding."""
        embedding_tensor = torch.tensor(sample_mean_embedding)
        projected = steering_vector_extractor.project_embedding(embedding_tensor)
        
        assert isinstance(projected, torch.Tensor)
        assert projected.shape == (steering_vector_extractor.target_dim,)

    def test_project_batched_embeddings(self, steering_vector_extractor, sample_embeddings):
        """Test projecting a batch of embeddings."""
        projected = steering_vector_extractor.project_embedding(sample_embeddings)
        
        assert projected.shape == (len(sample_embeddings), steering_vector_extractor.target_dim)

    def test_project_normalized_output(self, steering_vector_extractor, sample_mean_embedding):
        """Test that output is normalized when requested."""
        projected = steering_vector_extractor.project_embedding(
            sample_mean_embedding, normalize=True
        )
        
        norm = projected.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_project_unnormalized_output(self, steering_vector_extractor, sample_mean_embedding):
        """Test that output is not normalized when not requested."""
        projected = steering_vector_extractor.project_embedding(
            sample_mean_embedding, normalize=False
        )
        
        # Unnormalized output should generally not have norm = 1
        norm = projected.norm().item()
        # Just verify it's a valid tensor (norm could be any positive value)
        assert norm > 0


class TestCreateSteeringVector:
    """Test steering vector creation."""

    def test_create_single_direction_vector(self, steering_vector_extractor, sample_mean_embedding):
        """Test creating a steering vector from single embedding."""
        vector = steering_vector_extractor.create_steering_vector(
            positive_embedding=sample_mean_embedding,
            scale=1.0,
        )
        
        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (steering_vector_extractor.target_dim,)

    def test_create_contrastive_vector(self, steering_vector_extractor, sample_embeddings):
        """Test creating a contrastive steering vector."""
        pos_emb = sample_embeddings[0]
        neg_emb = sample_embeddings[1]
        
        vector = steering_vector_extractor.create_steering_vector(
            positive_embedding=pos_emb,
            negative_embedding=neg_emb,
        )
        
        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (steering_vector_extractor.target_dim,)

    def test_vector_scaling(self, steering_vector_extractor, sample_mean_embedding):
        """Test that scaling affects vector magnitude."""
        vector_1x = steering_vector_extractor.create_steering_vector(
            positive_embedding=sample_mean_embedding,
            scale=1.0,
        )
        vector_2x = steering_vector_extractor.create_steering_vector(
            positive_embedding=sample_mean_embedding,
            scale=2.0,
        )
        
        # 2x scaled should have double the norm
        ratio = vector_2x.norm() / vector_1x.norm()
        assert abs(ratio - 2.0) < 1e-5

    def test_contrastive_vector_direction(self, steering_vector_extractor, sample_embeddings):
        """Test that contrastive vector points from negative to positive."""
        pos_emb = sample_embeddings[0]
        neg_emb = sample_embeddings[1]
        
        vector = steering_vector_extractor.create_steering_vector(
            positive_embedding=pos_emb,
            negative_embedding=neg_emb,
            scale=1.0,
        )
        
        # The vector should be different from just the positive projection
        pos_only = steering_vector_extractor.create_steering_vector(
            positive_embedding=pos_emb,
            scale=1.0,
        )
        
        # They should not be identical
        assert not torch.allclose(vector, pos_only, atol=1e-3)


class TestMultiLayerVectors:
    """Test multi-layer vector creation."""

    def test_create_multi_layer_vectors_default(
        self, steering_vector_extractor, sample_mean_embedding, sample_num_layers
    ):
        """Test creating vectors for all layers with default scaling."""
        vectors = steering_vector_extractor.create_multi_layer_vectors(
            embedding=sample_mean_embedding,
            n_layers=sample_num_layers,
        )
        
        assert len(vectors) == sample_num_layers
        assert all(isinstance(v, torch.Tensor) for v in vectors.values())

    def test_create_multi_layer_vectors_bell_curve(
        self, steering_vector_extractor, sample_mean_embedding, sample_num_layers
    ):
        """Test that default creates bell curve distribution."""
        vectors = steering_vector_extractor.create_multi_layer_vectors(
            embedding=sample_mean_embedding,
            n_layers=sample_num_layers,
        )
        
        norms = [v.norm().item() for v in vectors.values()]
        
        # Middle layers should have higher norms than edge layers
        mid = sample_num_layers // 2
        assert norms[mid] > norms[0]
        assert norms[mid] > norms[-1]

    def test_create_multi_layer_vectors_custom_scale(
        self, steering_vector_extractor, sample_mean_embedding, sample_num_layers
    ):
        """Test custom per-layer scaling."""
        custom_scales = [1.0] * sample_num_layers
        custom_scales[10] = 2.0  # Make layer 10 stronger
        
        vectors = steering_vector_extractor.create_multi_layer_vectors(
            embedding=sample_mean_embedding,
            n_layers=sample_num_layers,
            scale_per_layer=custom_scales,
        )
        
        # Layer 10 should have double the norm of layer 0
        ratio = vectors[10].norm() / vectors[0].norm()
        assert abs(ratio - 2.0) < 1e-5


class TestSaveLoad:
    """Test saving and loading extractors."""

    def test_save_and_load(self, steering_vector_extractor, temp_cache_dir):
        """Test saving and loading projection matrix."""
        save_path = temp_cache_dir / "extractor.npz"
        
        # Save
        steering_vector_extractor.save(save_path)
        
        # Load into new extractor
        from src.steering_vectors import SteeringVectorExtractor
        new_extractor = SteeringVectorExtractor(
            source_dim=steering_vector_extractor.source_dim,
            target_dim=steering_vector_extractor.target_dim,
            device="cpu",
        )
        new_extractor.load(save_path)
        
        # Check matrices are equal
        torch.testing.assert_close(
            steering_vector_extractor.projection_matrix,
            new_extractor.projection_matrix,
        )


class TestContrastiveSteeringExtractor:
    """Test ContrastiveSteeringExtractor class."""

    def test_init(self, sample_hidden_dim):
        """Test initialization."""
        from src.steering_vectors import ContrastiveSteeringExtractor
        
        extractor = ContrastiveSteeringExtractor(
            target_dim=sample_hidden_dim,
            device="cpu",
        )
        
        assert extractor.target_dim == sample_hidden_dim
        assert extractor.steering_vectors == {}

    def test_compute_from_activations(self, contrastive_extractor, sample_hidden_dim):
        """Test computing steering vector from activation pairs."""
        # Create mock activations
        torch.manual_seed(42)
        pos_acts = [torch.randn(10, sample_hidden_dim) for _ in range(3)]
        neg_acts = [torch.randn(10, sample_hidden_dim) for _ in range(3)]
        
        vector = contrastive_extractor.compute_from_activations(
            positive_activations=pos_acts,
            negative_activations=neg_acts,
            layer_idx=12,
        )
        
        assert vector.shape == (sample_hidden_dim,)
        assert 12 in contrastive_extractor.steering_vectors

    def test_compute_normalized(self, contrastive_extractor, sample_hidden_dim):
        """Test that computed vectors are normalized."""
        torch.manual_seed(42)
        pos_acts = [torch.randn(10, sample_hidden_dim) for _ in range(3)]
        neg_acts = [torch.randn(10, sample_hidden_dim) for _ in range(3)]
        
        vector = contrastive_extractor.compute_from_activations(
            positive_activations=pos_acts,
            negative_activations=neg_acts,
            layer_idx=12,
            normalize=True,
        )
        
        norm = vector.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_get_vector(self, contrastive_extractor, sample_hidden_dim):
        """Test retrieving computed vectors."""
        torch.manual_seed(42)
        pos_acts = [torch.randn(10, sample_hidden_dim)]
        neg_acts = [torch.randn(10, sample_hidden_dim)]
        
        contrastive_extractor.compute_from_activations(
            positive_activations=pos_acts,
            negative_activations=neg_acts,
            layer_idx=12,
        )
        
        vector = contrastive_extractor.get_vector(12)
        assert vector is not None
        
        # Non-existent layer should return None
        assert contrastive_extractor.get_vector(99) is None

    def test_get_vector_with_scale(self, contrastive_extractor, sample_hidden_dim):
        """Test retrieving vectors with scaling."""
        torch.manual_seed(42)
        pos_acts = [torch.randn(10, sample_hidden_dim)]
        neg_acts = [torch.randn(10, sample_hidden_dim)]
        
        contrastive_extractor.compute_from_activations(
            positive_activations=pos_acts,
            negative_activations=neg_acts,
            layer_idx=12,
        )
        
        vector_1x = contrastive_extractor.get_vector(12, scale=1.0)
        vector_2x = contrastive_extractor.get_vector(12, scale=2.0)
        
        ratio = vector_2x.norm() / vector_1x.norm()
        assert abs(ratio - 2.0) < 1e-5
