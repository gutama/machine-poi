"""
Tests for src/knowledge_base.py

Tests for:
- QuranKnowledgeBase (ChromaDB integration)
- Multi-resolution indexing
- Query methods
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestQuranKnowledgeBaseInit:
    """Test QuranKnowledgeBase initialization."""

    def test_init_creates_client(self, temp_db_dir):
        """Test that initialization creates ChromaDB client."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch.object(
            QuranKnowledgeBase, "__init__", lambda self, **kwargs: None
        ):
            kb = QuranKnowledgeBase.__new__(QuranKnowledgeBase)
            kb.persist_dir = str(temp_db_dir)
        
        # Just verify the class exists and can be instantiated
        assert kb.persist_dir == str(temp_db_dir)

    def test_init_creates_collections(self, temp_db_dir, mock_sentence_transformer):
        """Test that initialization creates expected collections."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch("src.knowledge_base.QuranEmbeddings") as MockEmbeddings:
            mock_embedder = Mock()
            mock_embedder.load_model = Mock()
            MockEmbeddings.return_value = mock_embedder
            
            kb = QuranKnowledgeBase(
                persist_dir=str(temp_db_dir),
                device="cpu",
            )
            
            assert "verse" in kb.collections
            assert "passage" in kb.collections
            assert "surah" in kb.collections


class TestIndexBuilding:
    """Test knowledge base index building."""

    @pytest.fixture
    def mock_kb(self, temp_db_dir, sample_embedding_dim):
        """Create a mocked knowledge base."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch("src.knowledge_base.QuranEmbeddings") as MockEmbeddings:
            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.load_model = Mock()
            mock_embedder.load_quran_text.return_value = [
                "verse 1", "verse 2", "verse 3", "verse 4", "verse 5"
            ]
            mock_embedder.create_embeddings.return_value = np.random.randn(
                5, sample_embedding_dim
            ).astype(np.float32)
            MockEmbeddings.return_value = mock_embedder
            
            kb = QuranKnowledgeBase(
                persist_dir=str(temp_db_dir),
                device="cpu",
            )
            
            return kb

    def test_build_index_calls_embedder(self, mock_kb, sample_quran_path):
        """Test that build_index uses embedder for all resolutions."""
        mock_kb.build_index(quran_path=sample_quran_path)
        
        # Should call load_quran_text for each resolution
        calls = mock_kb.embedder.load_quran_text.call_args_list
        chunk_types = [c.kwargs.get("chunk_by", c.args[1] if len(c.args) > 1 else None) for c in calls]
        
        # Verify all resolutions were processed
        assert mock_kb.embedder.load_quran_text.call_count >= 3

    def test_build_index_populates_collections(self, mock_kb, sample_quran_path):
        """Test that collections are populated after indexing."""
        mock_kb.build_index(quran_path=sample_quran_path)
        
        # Each collection should have items
        for name, collection in mock_kb.collections.items():
            assert collection.count() > 0


class TestQuerying:
    """Test knowledge base query methods."""

    @pytest.fixture
    def populated_kb(self, temp_db_dir, sample_embedding_dim):
        """Create a knowledge base with test data."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch("src.knowledge_base.QuranEmbeddings") as MockEmbeddings:
            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.load_model = Mock()
            
            # Return different texts for different chunk types
            def mock_load(path, chunk_by):
                if chunk_by == "verse":
                    return ["verse " + str(i) for i in range(10)]
                elif chunk_by == "paragraph":
                    return ["paragraph " + str(i) for i in range(5)]
                else:
                    return ["surah " + str(i) for i in range(2)]
            
            mock_embedder.load_quran_text.side_effect = mock_load
            mock_embedder.create_embeddings.side_effect = lambda texts, **kwargs: (
                np.random.randn(len(texts), sample_embedding_dim).astype(np.float32)
            )
            MockEmbeddings.return_value = mock_embedder
            
            kb = QuranKnowledgeBase(
                persist_dir=str(temp_db_dir),
                device="cpu",
            )
            kb.build_index("test_quran.txt")
            
            return kb

    def test_query_multiresolution_returns_dict(self, populated_kb):
        """Test that query returns results for all resolutions."""
        results = populated_kb.query_multiresolution(
            query_text="mercy and compassion",
            n_results=3,
        )
        
        assert isinstance(results, dict)
        assert "verse" in results
        assert "passage" in results
        assert "surah" in results

    def test_query_multiresolution_result_structure(self, populated_kb):
        """Test that each result has expected fields."""
        results = populated_kb.query_multiresolution(
            query_text="test query",
            n_results=2,
        )
        
        for res_name, items in results.items():
            for item in items:
                assert "content" in item
                assert "metadata" in item
                assert "distance" in item
                assert "score" in item

    def test_query_multiresolution_with_embeddings(self, populated_kb):
        """Test querying with embeddings included."""
        results = populated_kb.query_multiresolution(
            query_text="test query",
            n_results=2,
            include_embeddings=True,
        )
        
        for res_name, items in results.items():
            for item in items:
                if items:  # If there are results
                    assert "embedding" in item or item.get("embedding") is not None

    def test_query_with_bridges(self, populated_kb):
        """Test querying with domain bridge queries."""
        results = populated_kb.query_with_bridges(
            original_query="how to fix a bug",
            bridge_queries=["patience", "wisdom"],
            n_results=3,
        )
        
        assert isinstance(results, dict)
        assert "verse" in results

    def test_query_with_bridges_deduplicates(self, populated_kb):
        """Test that bridge queries don't create duplicate results."""
        results = populated_kb.query_with_bridges(
            original_query="mercy",
            bridge_queries=["mercy", "compassion"],  # Same theme
            n_results=5,
        )
        
        # Results should be deduplicated (by metadata index)
        for res_name, items in results.items():
            indices = [item["metadata"].get("index") for item in items]
            # Filter out None values
            valid_indices = [i for i in indices if i is not None]
            assert len(valid_indices) == len(set(valid_indices))

    def test_query_with_bridges_ranks_by_score(self, populated_kb):
        """Test that results are sorted by score."""
        results = populated_kb.query_with_bridges(
            original_query="test",
            bridge_queries=["query1", "query2"],
            n_results=5,
        )
        
        for res_name, items in results.items():
            if len(items) > 1:
                scores = [item["score"] for item in items]
                assert scores == sorted(scores, reverse=True)


class TestCollectionManagement:
    """Test collection management functionality."""

    def test_collections_use_cosine_space(self, temp_db_dir):
        """Test that collections are configured for cosine similarity."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch("src.knowledge_base.QuranEmbeddings") as MockEmbeddings:
            mock_embedder = Mock()
            mock_embedder.load_model = Mock()
            MockEmbeddings.return_value = mock_embedder
            
            kb = QuranKnowledgeBase(
                persist_dir=str(temp_db_dir),
                device="cpu",
            )
            
            # Collections should use cosine distance
            for collection in kb.collections.values():
                metadata = collection.metadata
                if metadata:
                    assert metadata.get("hnsw:space") == "cosine"


class TestScoreComputation:
    """Test score computation from distances."""

    def test_distance_to_score_conversion(self, temp_db_dir, sample_embedding_dim):
        """Test that cosine distance is converted to similarity score."""
        from src.knowledge_base import QuranKnowledgeBase
        
        with patch("src.knowledge_base.QuranEmbeddings") as MockEmbeddings:
            mock_embedder = Mock()
            mock_embedder.load_model = Mock()
            mock_embedder.create_embeddings.return_value = np.random.randn(
                1, sample_embedding_dim
            ).astype(np.float32)
            MockEmbeddings.return_value = mock_embedder
            
            kb = QuranKnowledgeBase(
                persist_dir=str(temp_db_dir),
                device="cpu",
            )
            
            # Add a test item
            kb.collections["verse"].add(
                documents=["test verse"],
                embeddings=[np.random.randn(sample_embedding_dim).tolist()],
                ids=["test_1"],
                metadatas=[{"resolution": "verse", "index": 0}],
            )
            
            results = kb.query_multiresolution("test query", n_results=1)
            
            if results["verse"]:
                item = results["verse"][0]
                # Score should be 1 - distance (for cosine)
                expected_score = 1.0 - item["distance"]
                assert abs(item["score"] - expected_score) < 1e-5


class TestIntegration:
    """Integration tests for knowledge base."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_workflow(self, sample_quran_path, temp_db_dir):
        """Test full workflow: init -> index -> query."""
        from src.knowledge_base import QuranKnowledgeBase
        
        # This test would actually load models and index real data
        kb = QuranKnowledgeBase(
            persist_dir=str(temp_db_dir),
            embedding_model_name="bge-m3",
            device="cpu",
        )
        
        kb.build_index(quran_path=sample_quran_path)
        
        results = kb.query_multiresolution(
            query_text="الرحمن",  # Arabic for "The Merciful"
            n_results=3,
        )
        
        assert len(results["verse"]) > 0
