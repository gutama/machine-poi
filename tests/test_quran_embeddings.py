"""
Tests for src/quran_embeddings.py

Tests the QuranEmbeddings class for:
- Text loading and chunking
- Embedding creation
- Caching and loading
- Semantic clustering
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestQuranEmbeddingsInit:
    """Test QuranEmbeddings initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        
        assert embedder.model_name == "bge-m3"
        assert embedder.max_length == 512
        assert embedder.use_fp16 is True
        assert embedder.model is None  # Not loaded yet

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="multilingual-e5")
        
        assert embedder.model_name == "multilingual-e5"

    def test_init_device_selection_cpu(self):
        """Test explicit CPU device selection."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(device="cpu")
        
        assert embedder.device == "cpu"

    def test_supported_models_mapping(self):
        """Test that supported models are correctly mapped."""
        from src.quran_embeddings import QuranEmbeddings
        
        expected_models = ["bge-m3", "qwen-embedding", "bge-large-zh", "multilingual-e5"]
        
        for model in expected_models:
            assert model in QuranEmbeddings.SUPPORTED_MODELS


class TestTextLoading:
    """Test Quran text loading and chunking."""

    def test_load_quran_text_verse_chunking(self, sample_quran_path):
        """Test loading text with verse-level chunking."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="verse")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_load_quran_text_paragraph_chunking(self, sample_quran_path):
        """Test loading text with paragraph-level chunking."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="paragraph")
        
        # Paragraphs should be fewer than verses
        verse_chunks = embedder.load_quran_text(sample_quran_path, chunk_by="verse")
        assert len(chunks) <= len(verse_chunks)

    def test_load_quran_text_surah_chunking(self, sample_quran_path):
        """Test loading text with surah-level chunking."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        chunks = embedder.load_quran_text(sample_quran_path, chunk_by="surah")
        
        # Surahs should be fewest
        paragraph_chunks = embedder.load_quran_text(sample_quran_path, chunk_by="paragraph")
        assert len(chunks) <= len(paragraph_chunks)

    def test_load_quran_text_min_length_filter(self, sample_quran_path):
        """Test that short chunks are filtered out."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        
        # With high min_length, should filter out short verses
        chunks_strict = embedder.load_quran_text(
            sample_quran_path, chunk_by="verse", min_length=100
        )
        chunks_lenient = embedder.load_quran_text(
            sample_quran_path, chunk_by="verse", min_length=5
        )
        
        assert len(chunks_strict) <= len(chunks_lenient)

    def test_load_quran_text_file_not_found(self, tmp_path):
        """Test handling of missing file - falls back to default al-quran.txt."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        nonexistent = tmp_path / "nonexistent.txt"
        
        # The code falls back to al-quran.txt in the project root
        # So it should return chunks from that file instead of raising
        chunks = embedder.load_quran_text(nonexistent)
        
        # If fallback exists, we get chunks; otherwise would raise
        assert isinstance(chunks, list)


class TestEmbeddingCreation:
    """Test embedding creation functionality."""

    def test_create_embeddings_returns_correct_shape(
        self, mock_sentence_transformer, sample_embedding_dim
    ):
        """Test that embeddings have correct shape."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        texts = ["text one", "text two", "text three"]
        embeddings = embedder.create_embeddings(texts)
        
        assert embeddings.shape == (3, sample_embedding_dim)

    def test_create_embeddings_normalized(
        self, mock_sentence_transformer, sample_embedding_dim
    ):
        """Test that embeddings are L2 normalized."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        texts = ["sample text"]
        embeddings = embedder.create_embeddings(texts, normalize=True)
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_create_quran_embeddings_returns_dict(
        self, mock_sentence_transformer, sample_quran_path
    ):
        """Test that create_quran_embeddings returns proper dict."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            chunk_by="verse",
        )
        
        assert "embeddings" in result
        assert "texts" in result
        assert "mean_embedding" in result
        assert "model_name" in result
        assert "chunk_by" in result

    def test_create_quran_embeddings_mean_normalized(
        self, mock_sentence_transformer, sample_quran_path
    ):
        """Test that mean embedding is normalized."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            chunk_by="verse",
        )
        
        mean_norm = np.linalg.norm(result["mean_embedding"])
        assert abs(mean_norm - 1.0) < 1e-5


class TestCaching:
    """Test embedding caching and loading."""

    def test_save_and_load_embeddings(
        self, mock_sentence_transformer, sample_quran_path, temp_cache_dir
    ):
        """Test that embeddings can be saved and loaded."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        save_path = temp_cache_dir / "test_embeddings.npz"
        
        # Create and save
        result = embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            save_path=save_path,
        )
        
        # Load
        loaded = embedder.load_cached_embeddings(save_path)
        
        np.testing.assert_array_almost_equal(
            result["embeddings"], loaded["embeddings"]
        )
        np.testing.assert_array_almost_equal(
            result["mean_embedding"], loaded["mean_embedding"]
        )

    def test_texts_saved_separately(
        self, mock_sentence_transformer, sample_quran_path, temp_cache_dir
    ):
        """Test that texts are saved in a separate file."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        embedder.model = mock_sentence_transformer
        
        save_path = temp_cache_dir / "test_embeddings.npz"
        
        embedder.create_quran_embeddings(
            file_path=sample_quran_path,
            save_path=save_path,
        )
        
        texts_path = save_path.with_suffix(".texts.txt")
        assert texts_path.exists()


class TestSemanticClustering:
    """Test semantic clustering functionality."""

    def test_get_semantic_clusters(self, sample_embeddings):
        """Test that clustering returns correct structure."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        n_clusters = 3
        
        result = embedder.get_semantic_clusters(sample_embeddings, n_clusters=n_clusters)
        
        assert "centers" in result
        assert "labels" in result
        assert "n_clusters" in result
        assert result["n_clusters"] == n_clusters
        assert len(result["labels"]) == len(sample_embeddings)

    def test_cluster_centers_normalized(self, sample_embeddings):
        """Test that cluster centers are normalized."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings()
        result = embedder.get_semantic_clusters(sample_embeddings, n_clusters=3)
        
        for center in result["centers"]:
            norm = np.linalg.norm(center)
            assert abs(norm - 1.0) < 1e-4


class TestModelLoading:
    """Test model loading (integration-style, may be slow)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_model_bge_m3(self):
        """Test loading the bge-m3 model."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="bge-m3", device="cpu")
        embedder.load_model()
        
        assert embedder.model is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_encode_with_real_model(self):
        """Test encoding with a real model."""
        from src.quran_embeddings import QuranEmbeddings
        
        embedder = QuranEmbeddings(model_name="bge-m3", device="cpu")
        embedder.load_model()
        
        texts = ["بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"]
        embeddings = embedder.create_embeddings(texts)
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0
