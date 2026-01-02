"""
Tests for LightRAG Knowledge Graph Integration
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_quran_texts():
    """Sample Quranic texts for testing."""
    return [
        "Indeed, with hardship comes ease. Indeed, with hardship comes ease.",
        "And We have certainly made the Quran easy for remembrance.",
        "And whoever relies upon Allah - then He is sufficient for him.",
        "So remember Me; I will remember you.",
        "Indeed, prayer prohibits immorality and wrongdoing.",
    ]


@pytest.fixture
def mock_embedding_func():
    """Create a mock async embedding function."""
    async def embed(texts):
        return [np.random.randn(384).tolist() for _ in texts]
    return embed


@pytest.fixture
def mock_llm_func():
    """Create a mock async LLM function."""
    async def complete(prompt, **kwargs):
        return '{"entities": [], "relationships": []}'
    return complete


@pytest.fixture
def sample_embedding_dim():
    """Standard embedding dimension for tests."""
    return 384


# =============================================================================
# Test QuranLightRAG Adapter
# =============================================================================

class TestQuranLightRAG:
    """Tests for QuranLightRAG adapter."""

    def test_entity_types_defined(self):
        """Test that entity types are properly defined."""
        from src.lightrag_adapter import QuranLightRAG
        
        expected_types = [
            "prophet", "angel", "virtue", "command", "concept",
            "event", "place", "group", "practice", "scripture"
        ]
        assert QuranLightRAG.ENTITY_TYPES == expected_types

    def test_init_creates_working_dir(self, tmp_path, mock_embedding_func, mock_llm_func):
        """Test that initialization creates the working directory."""
        from src.lightrag_adapter import QuranLightRAG
        
        working_dir = tmp_path / "test_lightrag"
        rag = QuranLightRAG(
            working_dir=str(working_dir),
            embedding_func=mock_embedding_func,
            embedding_dim=384,
            llm_func=mock_llm_func,
        )
        
        assert working_dir.exists()
        assert rag.embedding_dim == 384
        assert rag._initialized is False

    def test_requires_embedding_func(self, tmp_path):
        """Test that initialization without embedding_func raises error."""
        from src.lightrag_adapter import QuranLightRAG
        
        rag = QuranLightRAG(
            working_dir=str(tmp_path / "test_rag"),
            embedding_func=None,
            llm_func=Mock(),
        )
        
        with pytest.raises(ValueError, match="embedding_func must be provided"):
            asyncio.get_event_loop().run_until_complete(rag.initialize())


# =============================================================================
# Test GraphBridgeGenerator
# =============================================================================

class TestGraphBridgeGenerator:
    """Tests for graph-based bridge generation."""

    def test_term_to_entity_mapping_exists(self):
        """Test that TERM_TO_ENTITY mappings are defined."""
        from src.graph_bridge import GraphBridgeGenerator
        
        # Check key mappings exist
        assert "debug" in GraphBridgeGenerator.TERM_TO_ENTITY
        assert "stress" in GraphBridgeGenerator.TERM_TO_ENTITY
        assert "success" in GraphBridgeGenerator.TERM_TO_ENTITY
        
        # Check mapped entities are list of strings
        assert isinstance(GraphBridgeGenerator.TERM_TO_ENTITY["debug"], list)
        assert "patience" in GraphBridgeGenerator.TERM_TO_ENTITY["debug"]

    def test_extract_query_terms(self):
        """Test extraction of terms from user queries."""
        from src.graph_bridge import GraphBridgeGenerator
        
        # Create mock LightRAG
        mock_lightrag = MagicMock()
        gen = GraphBridgeGenerator(mock_lightrag)
        
        # Test term extraction
        terms = gen._extract_query_terms("How do I debug my code under stress?")
        assert "debug" in terms
        assert "stress" in terms
        
        # Test with no matching terms
        terms = gen._extract_query_terms("Hello world")
        assert len(terms) == 0

    def test_bridge_relationships_defined(self):
        """Test that bridge relationships are defined."""
        from src.graph_bridge import GraphBridgeGenerator
        
        expected_rels = {
            "exemplifies", "teaches", "leads_to", "requires",
            "contrasts_with", "manifests_as", "is_aspect_of", "practiced_by"
        }
        assert GraphBridgeGenerator.BRIDGE_RELATIONSHIPS == expected_rels


# =============================================================================
# Test HybridQuranKnowledgeBase
# =============================================================================

class TestHybridQuranKnowledgeBase:
    """Tests for hybrid knowledge base."""

    def test_query_modes_accepted(self):
        """Test that all query modes are valid."""
        from src.hybrid_knowledge_base import HybridQuranKnowledgeBase
        
        kb = HybridQuranKnowledgeBase()
        
        # These are the valid modes - just test the object initializes
        assert kb.vector_persist_dir == "quran_db"
        assert kb.graph_working_dir == "quran_lightrag"
        assert kb._initialized is False

    def test_hybrid_query_result_structure(self):
        """Test HybridQueryResult dataclass structure."""
        from src.hybrid_knowledge_base import HybridQueryResult
        
        result = HybridQueryResult(
            vector_results={"verse": []},
            graph_answer="Test answer",
            graph_entities=["entity1"],
            graph_relationships=[],
            bridges=["patience"],
            bridge_confidence={"patience": 0.8},
            query_mode="hybrid",
            fusion_strategy="interleave",
        )
        
        assert result.graph_answer == "Test answer"
        assert result.query_mode == "hybrid"
        assert "patience" in result.bridges


# =============================================================================
# Test LLM Adapters
# =============================================================================

class TestLLMAdapters:
    """Tests for LLM function adapters."""

    def test_ollama_adapter_creation(self):
        """Test Ollama adapter can be created."""
        from src.llm_adapters import create_ollama_adapter
        
        adapter = create_ollama_adapter(model_name="qwen2.5:7b")
        assert callable(adapter)

    def test_local_llm_adapter_creation(self):
        """Test local LLM adapter can be created."""
        from src.llm_adapters import create_local_llm_adapter
        
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Generated text")
        
        adapter = create_local_llm_adapter(mock_llm)
        assert callable(adapter)


# =============================================================================
# Test Entity Extraction Prompts
# =============================================================================

class TestEntityExtractionPrompts:
    """Tests for entity extraction prompts."""

    def test_prompts_defined(self):
        """Test that all prompts are defined."""
        from src.prompts.entity_extraction import (
            ENTITY_EXTRACTION_PROMPT,
            RELATIONSHIP_ENHANCEMENT_PROMPT,
            THEME_BRIDGING_PROMPT,
        )
        
        assert "{text}" in ENTITY_EXTRACTION_PROMPT
        assert "{entities}" in RELATIONSHIP_ENHANCEMENT_PROMPT
        assert "{query}" in THEME_BRIDGING_PROMPT

    def test_entity_types_in_prompt(self):
        """Test that entity types are mentioned in extraction prompt."""
        from src.prompts.entity_extraction import ENTITY_EXTRACTION_PROMPT
        
        # Key entity types should be mentioned
        assert "PROPHET" in ENTITY_EXTRACTION_PROMPT
        assert "VIRTUE" in ENTITY_EXTRACTION_PROMPT
        assert "COMMAND" in ENTITY_EXTRACTION_PROMPT


# =============================================================================
# Test Steerer Integration
# =============================================================================

class TestSteererIntegration:
    """Tests for QuranSteerer graph integration."""

    def test_steerer_has_graph_params(self):
        """Test that QuranSteerer accepts graph parameters."""
        from src.steerer import QuranSteerer
        
        # These should not raise
        steerer = QuranSteerer(
            use_graph_kb=True,
            llm_func=AsyncMock(),
        )
        
        assert steerer.use_graph_kb is True
        assert steerer._llm_func is not None

    def test_steerer_has_hybrid_kb_attribute(self):
        """Test that QuranSteerer has hybrid_kb attribute."""
        from src.steerer import QuranSteerer
        
        steerer = QuranSteerer()
        
        assert hasattr(steerer, 'hybrid_kb')
        assert hasattr(steerer, 'graph_bridge_generator')
        assert steerer.hybrid_kb is None  # Not initialized yet

    def test_generate_domain_bridges_has_use_graph_param(self):
        """Test that generate_domain_bridges accepts use_graph parameter."""
        from src.steerer import QuranSteerer
        import inspect
        
        sig = inspect.signature(QuranSteerer.generate_domain_bridges)
        params = list(sig.parameters.keys())
        
        assert 'use_graph' in params


# =============================================================================
# Test Config Updates
# =============================================================================

class TestConfigUpdates:
    """Tests for config.py updates."""

    def test_steering_defaults_has_graph_settings(self):
        """Test that SteeringDefaults has graph-related settings."""
        from config import STEERING_DEFAULTS
        
        assert hasattr(STEERING_DEFAULTS, 'graph_chunk_size')
        assert hasattr(STEERING_DEFAULTS, 'graph_top_k')
        assert hasattr(STEERING_DEFAULTS, 'use_graph_bridges')
        assert hasattr(STEERING_DEFAULTS, 'hybrid_query_mode')

    def test_lightrag_storage_backends_defined(self):
        """Test that LIGHTRAG_STORAGE_BACKENDS is defined."""
        from config import LIGHTRAG_STORAGE_BACKENDS
        
        assert "default" in LIGHTRAG_STORAGE_BACKENDS
        assert "production" in LIGHTRAG_STORAGE_BACKENDS
        
        # Check default backend structure
        default = LIGHTRAG_STORAGE_BACKENDS["default"]
        assert "kv_storage" in default
        assert "vector_storage" in default
        assert "graph_storage" in default
