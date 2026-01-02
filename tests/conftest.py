"""
Pytest fixtures and configuration for Machine-POI tests.

Provides shared fixtures for:
- Mock models (to avoid loading real LLMs in tests)
- Sample data (Quran text, embeddings)
- Temporary directories
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_quran_path(tmp_path):
    """Create a temporary Quran text file with sample verses."""
    sample_verses = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "الرَّحْمَٰنِ الرَّحِيمِ",
        "مَالِكِ يَوْمِ الدِّينِ",
        "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
        # Add more verses for paragraph/surah chunking tests
        "الم ذَٰلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ هُدًى لِّلْمُتَّقِينَ",
        "الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ وَيُقِيمُونَ الصَّلَاةَ",
        "وَمِمَّا رَزَقْنَاهُمْ يُنفِقُونَ",
    ]
    
    quran_file = tmp_path / "test_quran.txt"
    quran_file.write_text("\n".join(sample_verses), encoding="utf-8")
    return quran_file


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_db_dir(tmp_path):
    """Create a temporary database directory."""
    db_dir = tmp_path / "quran_db"
    db_dir.mkdir()
    return db_dir


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_embedding_dim():
    """Standard embedding dimension for tests."""
    return 1024


@pytest.fixture
def sample_hidden_dim():
    """Standard LLM hidden dimension for tests."""
    return 896


@pytest.fixture
def sample_num_layers():
    """Standard number of LLM layers for tests."""
    return 24


@pytest.fixture
def sample_embeddings(sample_embedding_dim):
    """Generate sample embeddings."""
    np.random.seed(42)
    n_samples = 10
    embeddings = np.random.randn(n_samples, sample_embedding_dim).astype(np.float32)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def sample_mean_embedding(sample_embeddings):
    """Generate a mean embedding from samples."""
    mean = sample_embeddings.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    return mean


@pytest.fixture
def sample_steering_vector(sample_hidden_dim):
    """Generate a sample steering vector."""
    torch.manual_seed(42)
    vector = torch.randn(sample_hidden_dim)
    vector = vector / vector.norm()
    return vector


@pytest.fixture
def sample_hidden_states(sample_hidden_dim):
    """Generate sample hidden states [batch, seq_len, hidden_dim]."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 16
    return torch.randn(batch_size, seq_len, sample_hidden_dim)


# =============================================================================
# Mock Model Fixtures
# =============================================================================

@pytest.fixture
def mock_sentence_transformer(sample_embedding_dim):
    """Create a mock SentenceTransformer model."""
    mock_model = Mock()
    
    def mock_encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        np.random.seed(hash(str(texts)) % 2**32)
        embeddings = np.random.randn(n, sample_embedding_dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    mock_model.encode = mock_encode
    mock_model.half = Mock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_llm_model(sample_hidden_dim, sample_num_layers):
    """Create a mock HuggingFace LLM model."""
    mock_model = MagicMock()
    
    # Config
    mock_model.config.hidden_size = sample_hidden_dim
    mock_model.config.num_hidden_layers = sample_num_layers
    
    # Model layers (for hook registration)
    mock_layers = MagicMock()
    mock_model.model.layers = [MagicMock() for _ in range(sample_num_layers)]
    
    # Generate method
    def mock_generate(input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = kwargs.get("max_new_tokens", 50)
        return torch.randint(0, 50000, (batch_size, input_ids.shape[1] + seq_len))
    
    mock_model.generate = mock_generate
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=mock_model)
    
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    mock_tok = Mock()
    mock_tok.pad_token = "<pad>"
    mock_tok.eos_token = "</s>"
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    
    def mock_call(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        seq_len = max(len(t.split()) for t in text) + 5
        return {
            "input_ids": torch.randint(0, 50000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
        }
    
    mock_tok.__call__ = mock_call
    mock_tok.encode = lambda x, **kwargs: list(range(len(x.split()) + 5))
    mock_tok.decode = lambda x, **kwargs: "Generated mock output text."
    mock_tok.batch_decode = lambda x, **kwargs: ["Generated mock output text."] * len(x)
    
    return mock_tok


# =============================================================================
# Component Fixtures (with mocks injected)
# =============================================================================

@pytest.fixture
def steering_vector_extractor(sample_embedding_dim, sample_hidden_dim):
    """Create a SteeringVectorExtractor with test dimensions."""
    from src.steering_vectors import SteeringVectorExtractor
    return SteeringVectorExtractor(
        source_dim=sample_embedding_dim,
        target_dim=sample_hidden_dim,
        projection_type="random",
        device="cpu",
    )


@pytest.fixture
def contrastive_extractor(sample_hidden_dim):
    """Create a ContrastiveSteeringExtractor."""
    from src.steering_vectors import ContrastiveSteeringExtractor
    return ContrastiveSteeringExtractor(
        target_dim=sample_hidden_dim,
        device="cpu",
    )


@pytest.fixture
def activation_hook(sample_steering_vector):
    """Create an ActivationHook for testing."""
    from src.llm_wrapper import ActivationHook
    return ActivationHook(
        layer_idx=12,
        steering_vector=sample_steering_vector,
        coefficient=0.5,
        injection_mode="clamp",
    )


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# =============================================================================
# Skip Conditions
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
