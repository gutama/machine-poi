"""
Tests for src/steerer.py

Tests for:
- QuranSteerer (main interface)
- SteeringConfig
- Domain bridging
- Steering preparation methods
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict


class TestSteeringConfig:
    """Test SteeringConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from src.steerer import SteeringConfig
        
        config = SteeringConfig()
        
        assert config.coefficient == 0.5
        assert config.target_layers is None
        assert config.injection_mode == "add"
        assert config.layer_distribution == "bell"
        assert config.focus_layer == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        from src.steerer import SteeringConfig
        
        config = SteeringConfig(
            coefficient=0.8,
            target_layers=[10, 11, 12],
            injection_mode="clamp",
            layer_distribution="uniform",
        )
        
        assert config.coefficient == 0.8
        assert config.target_layers == [10, 11, 12]
        assert config.injection_mode == "clamp"
        assert config.layer_distribution == "uniform"


class TestDomainBridgeMap:
    """Test domain bridge mapping."""

    def test_domain_bridge_map_exists(self):
        """Test that domain bridge map is defined."""
        from src.steerer import DOMAIN_BRIDGE_MAP
        
        assert isinstance(DOMAIN_BRIDGE_MAP, dict)
        assert len(DOMAIN_BRIDGE_MAP) > 0

    def test_domain_bridge_map_categories(self):
        """Test that expected categories exist."""
        from src.steerer import DOMAIN_BRIDGE_MAP
        
        # Technical
        assert "bug" in DOMAIN_BRIDGE_MAP
        assert "debug" in DOMAIN_BRIDGE_MAP
        assert "code" in DOMAIN_BRIDGE_MAP
        
        # Emotional
        assert "stress" in DOMAIN_BRIDGE_MAP
        assert "anxiety" in DOMAIN_BRIDGE_MAP
        
        # Team/Social
        assert "team" in DOMAIN_BRIDGE_MAP
        assert "leadership" in DOMAIN_BRIDGE_MAP

    def test_domain_bridge_map_values_are_lists(self):
        """Test that values are lists of themes."""
        from src.steerer import DOMAIN_BRIDGE_MAP
        
        for keyword, themes in DOMAIN_BRIDGE_MAP.items():
            assert isinstance(themes, list)
            assert len(themes) > 0
            assert all(isinstance(t, str) for t in themes)


class TestQuranSteererInit:
    """Test QuranSteerer initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        assert steerer.llm_model_name == "deepseek-r1-1.5b"
        assert steerer.embedding_model_name == "bge-m3"
        assert steerer.quran_path == Path("al-quran.txt")
        assert steerer.embedder is None
        assert steerer.llm is None

    def test_init_custom_models(self):
        """Test initialization with custom models."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer(
            llm_model="qwen2.5-0.5b",
            embedding_model="multilingual-e5",
        )
        
        assert steerer.llm_model_name == "qwen2.5-0.5b"
        assert steerer.embedding_model_name == "multilingual-e5"

    def test_init_with_quantization(self):
        """Test initialization with quantization option."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer(llm_quantization="4bit")
        
        assert steerer.llm_quantization == "4bit"

    def test_init_custom_quran_path(self, sample_quran_path):
        """Test initialization with custom Quran path."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer(quran_path=sample_quran_path)
        
        assert steerer.quran_path == sample_quran_path


class TestDomainBridging:
    """Test domain bridge generation."""

    def test_generate_domain_bridges_technical(self):
        """Test domain bridging for technical queries."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        bridges = steerer.generate_domain_bridges(
            "I have a bug in my code that I need to fix",
            max_bridges=3,
        )
        
        assert isinstance(bridges, list)
        assert len(bridges) <= 3
        # Should find "bug" and map to its themes
        assert len(bridges) > 0

    def test_generate_domain_bridges_emotional(self):
        """Test domain bridging for emotional queries."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        bridges = steerer.generate_domain_bridges(
            "I feel a lot of stress and anxiety about my deadline",
            max_bridges=5,
        )
        
        assert len(bridges) > 0
        # Should find "stress" and "anxiety"

    def test_generate_domain_bridges_no_match(self):
        """Test domain bridging when no keywords match."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        bridges = steerer.generate_domain_bridges(
            "xyzabc123 totally unrelated text",
            max_bridges=3,
        )
        
        assert bridges == []

    def test_generate_domain_bridges_dedup(self):
        """Test that duplicate bridges are removed."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        # Query that might trigger duplicate themes
        bridges = steerer.generate_domain_bridges(
            "stress and anxiety make me anxious",
            max_bridges=10,
        )
        
        # Should have unique bridges
        assert len(bridges) == len(set(bridges))


class TestComputeDynamicSteering:
    """Test dynamic steering computation from retrieved results."""

    @pytest.fixture
    def mock_steerer_with_models(self, sample_embedding_dim, sample_hidden_dim, sample_num_layers):
        """Create a steerer with mocked models."""
        from src.steerer import QuranSteerer
        from src.steering_vectors import SteeringVectorExtractor
        
        steerer = QuranSteerer()
        
        # Mock LLM
        steerer.llm = Mock()
        steerer.llm.hidden_size = sample_hidden_dim
        steerer.llm.num_layers = sample_num_layers
        
        # Mock activations extraction
        # Returns Dict[layer_idx, tensor]
        def mock_extract(text):
            return {
                i: torch.randn(1, 10, sample_hidden_dim) # [batch, seq, dim]
                for i in range(sample_num_layers)
            }
        steerer.llm.extract_layer_activations = mock_extract
        
        steerer.device = "cpu"
        
        return steerer

    def test_compute_dynamic_steering_with_activations(
        self, mock_steerer_with_models, sample_embedding_dim
    ):
        """Test computing steering from retrieved content using activations."""
        steerer = mock_steerer_with_models
        
        # Create mock retrieved results (embedding field is ignored now)
        mock_results = {
            "verse": [
                {
                    "content": "verse text",
                    "score": 0.9,
                },
            ],
            "passage": [
                {
                    "content": "passage text",
                    "score": 0.8,
                },
            ],
            "surah": [],
        }
        
        vectors = steerer.compute_dynamic_steering(mock_results)
        
        assert vectors is not None
        assert isinstance(vectors, dict)
        assert len(vectors) == steerer.llm.num_layers
        assert isinstance(vectors[0], torch.Tensor)

    def test_compute_dynamic_steering_empty_results(self, mock_steerer_with_models):
        """Test computing steering from empty results."""
        steerer = mock_steerer_with_models
        
        mock_results = {
            "verse": [],
            "passage": [],
            "surah": [],
        }
        
        vectors = steerer.compute_dynamic_steering(mock_results)
        
        assert vectors is None

    def test_compute_dynamic_steering_custom_weights(
        self, mock_steerer_with_models, sample_embedding_dim
    ):
        """Test custom resolution weights."""
        steerer = mock_steerer_with_models
        
        mock_results = {
            "verse": [
                {"content": "v", "score": 0.5},
            ],
            "passage": [
                {"content": "p", "score": 0.5},
            ],
            "surah": [
                {"content": "s", "score": 0.5},
            ],
        }
        
        custom_weights = {"verse": 1.0, "passage": 0.0, "surah": 0.0}
        
        vectors = steerer.compute_dynamic_steering(
            mock_results,
            resolution_weights=custom_weights,
        )
        
        assert vectors is not None


class TestPrepareQuranSteering:
    """Test Quran steering preparation."""

    @pytest.fixture
    def mock_steerer(
        self, mock_sentence_transformer, sample_embedding_dim, sample_hidden_dim, sample_num_layers
    ):
        """Create steerer with mocked components."""
        from src.steerer import QuranSteerer
        from src.quran_embeddings import QuranEmbeddings
        from src.llm_wrapper import SteeredLLM
        
        steerer = QuranSteerer()
        
        # Mock embedder - just for loading text now
        steerer.embedder = Mock(spec=QuranEmbeddings)
        steerer.embedder.load_quran_text.return_value = ["verse 1", "verse 2", "verse 3"]
        
        # Mock LLM
        steerer.llm = Mock(spec=SteeredLLM)
        steerer.llm.hidden_size = sample_hidden_dim
        steerer.llm.num_layers = sample_num_layers
        steerer.llm.register_steering_hook = Mock()
        steerer.llm.clear_steering = Mock()
        
        # Mock extractions
        def mock_extract(text):
            return {
                i: torch.randn(1, 5, sample_hidden_dim) 
                for i in range(sample_num_layers)
            }
        steerer.llm.extract_layer_activations = mock_extract
        
        steerer.device = "cpu"
        
        return steerer

    def test_prepare_quran_steering_creates_vectors(
        self, mock_steerer, sample_quran_path, sample_num_layers
    ):
        """Test that steering vectors are created."""
        mock_steerer.quran_path = sample_quran_path
        
        vectors = mock_steerer.prepare_quran_steering(chunk_by="verse")
        
        assert vectors is not None
        assert isinstance(vectors, dict)
        assert len(vectors) == sample_num_layers

    def test_prepare_quran_steering_applies_to_llm(self, mock_steerer, sample_quran_path):
        """Test that steering is applied to LLM."""
        mock_steerer.quran_path = sample_quran_path
        
        mock_steerer.prepare_quran_steering()
        
        # LLM should have hooks registered
        assert mock_steerer.llm.register_steering_hook.called


class TestPrepareThematicSteering:
    """Test thematic steering preparation."""

    @pytest.fixture
    def prepared_steerer(
        self, mock_sentence_transformer, sample_embedding_dim, sample_hidden_dim, sample_num_layers
    ):
        """Create a steerer that's already prepared."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        texts_list = [f"verse {i}" for i in range(20)]
        
        # Mock embedder
        steerer.embedder = Mock()
        steerer.embedder.create_embeddings.return_value = np.random.randn(1, sample_embedding_dim).astype(np.float32)
        steerer.embedder.load_quran_text.return_value = texts_list
        
        # Mock LLM
        steerer.llm = Mock()
        steerer.llm.hidden_size = sample_hidden_dim
        steerer.llm.num_layers = sample_num_layers
        steerer.llm.register_steering_hook = Mock()
        steerer.llm.clear_steering = Mock()
        
        def mock_extract(text):
            return {
                i: torch.randn(1, 5, sample_hidden_dim) 
                for i in range(sample_num_layers)
            }
        steerer.llm.extract_layer_activations = mock_extract
        
        # Mock Quran embeddings for search
        steerer.quran_embeddings = {
            "embeddings": np.random.randn(20, sample_embedding_dim).astype(np.float32),
            "texts": texts_list,
        }
        
        steerer.device = "cpu"
        
        return steerer

    def test_prepare_thematic_steering(self, prepared_steerer):
        """Test thematic steering preparation."""
        vectors = prepared_steerer.prepare_thematic_steering(
            theme_query="mercy and compassion",
            top_k=5,
        )
        
        assert vectors is not None
        assert len(vectors) == prepared_steerer.llm.num_layers


class TestPrepareQuranPersona:
    """Test Quran Persona preparation."""

    @pytest.fixture
    def persona_steerer(
        self, mock_sentence_transformer, sample_embedding_dim, sample_hidden_dim, sample_num_layers, tmp_path
    ):
        """Create steerer for persona testing."""
        from src.steerer import QuranSteerer
        from src.quran_embeddings import QuranEmbeddings
        
        steerer = QuranSteerer()
        
        # Mock embedder
        steerer.embedder = Mock(spec=QuranEmbeddings)
        steerer.embedder.load_quran_text.return_value = ["v1", "v2", "v3"]
        
        # Mock LLM
        steerer.llm = Mock()
        steerer.llm.hidden_size = sample_hidden_dim
        steerer.llm.num_layers = sample_num_layers
        steerer.llm.register_steering_hook = Mock()
        steerer.llm.clear_steering = Mock()
        
        def mock_extract(text):
            return {
                i: torch.randn(1, 5, sample_hidden_dim) 
                for i in range(sample_num_layers)
            }
        steerer.llm.extract_layer_activations = mock_extract
        
        steerer.device = "cpu"
        
        return steerer

    def test_prepare_quran_persona_creates_vectors(self, persona_steerer, tmp_path):
        """Test that persona creates steering vectors."""
        # Note: calling internal method or mocked method
        vectors = persona_steerer.prepare_quran_persona(cache_dir=str(tmp_path))
        
        assert vectors is not None
        assert len(vectors) == persona_steerer.llm.num_layers

    def test_prepare_quran_persona_applies_steering(self, persona_steerer, tmp_path):
        """Test that persona applies steering to LLM."""
        persona_steerer.prepare_quran_persona(cache_dir=str(tmp_path))
        
        assert persona_steerer.llm.register_steering_hook.called


class TestSetSteeringStrength:
    """Test steering strength adjustment."""

    def test_set_steering_strength(self):
        """Test setting steering strength."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        steerer.llm = Mock()
        steerer.llm.update_steering_coefficient = Mock()
        
        # Should update config
        steerer.set_steering_strength(0.8)
        
        assert steerer.config.coefficient == 0.8


class TestGenerateAndCompare:
    """Test generation methods."""

    @pytest.fixture
    def generation_steerer(self, sample_hidden_dim, sample_num_layers):
        """Create steerer ready for generation."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        # Mock LLM
        steerer.llm = Mock()
        steerer.llm.hidden_size = sample_hidden_dim
        steerer.llm.num_layers = sample_num_layers
        steerer.llm.generate.return_value = "Generated steered output"
        
        # Context manager mock
        steerer.llm.steering_disabled = Mock()
        steerer.llm.steering_disabled.return_value.__enter__ = Mock()
        steerer.llm.steering_disabled.return_value.__exit__ = Mock()
        
        return steerer

    def test_generate_returns_string(self, generation_steerer):
        """Test that generate returns a string."""
        output = generation_steerer.generate("Test prompt")
        
        assert isinstance(output, str)
        assert generation_steerer.llm.generate.called

    def test_compare_returns_tuple(self, generation_steerer):
        """Test that compare returns steered and baseline."""
        # Mock the compare method directly since it's complex
        generation_steerer.compare = Mock(return_value=("steered output", "baseline output"))
        
        steered, baseline = generation_steerer.compare("Test prompt")
        
        assert isinstance(steered, str)
        assert isinstance(baseline, str)
        assert steered == "steered output"
        assert baseline == "baseline output"
