"""
Quran Text Embeddings Module

Creates semantic embeddings from Quranic verses using models like:
- Qwen3-Embedding-8B (Alibaba-NLP/gte-Qwen2-7B-instruct or similar)
- BGE-M3 (BAAI/bge-m3)
"""

import gc
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, List, Dict, Literal, Any
import numpy as np
import torch

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STEERING_DEFAULTS

# Setup module logger
logger = logging.getLogger("machine_poi.quran_embeddings")


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class QuranFileError(EmbeddingError):
    """Raised when Quran file is invalid or missing."""
    pass


class LRUCache:
    """Simple LRU cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> None:
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)



class QuranEmbeddings:
    """
    Creates and manages embeddings from Quran text using various embedding models.

    Supports:
    - BAAI/bge-m3: Multilingual embeddings with strong Arabic support
    - Alibaba-NLP/gte-Qwen2-7B-instruct: Large-scale instruction-tuned embeddings
    """

    SUPPORTED_MODELS = {
        "bge-m3": "BAAI/bge-m3",
        "qwen-embedding": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "multilingual-e5": "intfloat/multilingual-e5-large-instruct",
        "multilingual-e5-large": "intfloat/multilingual-e5-large",
        "paraphrase-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }

    # Standard verse counts for all 114 Surahs
    SURAH_VERSE_COUNTS = [
        7, 286, 200, 176, 120, 165, 206, 75, 129, 109,
        123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
        112, 78, 118, 64, 77, 227, 93, 88, 69, 60,
        34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
        54, 53, 89, 59, 37, 35, 38, 29, 18, 45,
        60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
        14, 11, 11, 18, 12, 12, 30, 52, 52, 44,
        28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
        29, 19, 36, 25, 22, 17, 19, 26, 30, 20,
        15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
        11, 8, 3, 9, 5, 4, 7, 3, 6, 3,
        5, 4, 5, 6
    ]

    def __init__(
        self,
        model_name: str = "bge-m3",
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize the Quran embeddings generator.

        Args:
            model_name: Name of the embedding model to use
            device: Device to run on (cuda/cpu/mps)
            use_fp16: Use half precision for memory efficiency
            max_length: Maximum sequence length for embeddings
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._embeddings_cache = LRUCache(max_size=STEERING_DEFAULTS.max_embedding_cache_size)

    def load_model(self) -> None:
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        model_path = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        logger.info(f"Loading embedding model: {model_path}")

        # For larger models, use specific loading strategies
        if "qwen" in model_path.lower() or "7b" in model_path.lower():
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
        else:
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
            )

        if self.use_fp16 and self.device != "cpu":
            self.model = self.model.half()

        logger.info(f"Model loaded on {self.device}")

    def load_quran_text(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Literal["verse", "surah", "paragraph"] = "verse",
        min_length: Optional[int] = None,
    ) -> List[str]:
        """
        Load and chunk the Quran text.

        Args:
            file_path: Path to the Quran text file
            chunk_by: How to split the text (verse, surah, paragraph)
            min_length: Minimum character length for a chunk (default from config)

        Returns:
            List of text chunks
            
        Raises:
            QuranFileError: If the file cannot be found or read
        """
        if min_length is None:
            min_length = STEERING_DEFAULTS.min_chunk_length
            
        file_path = Path(file_path)
        if not file_path.exists():
            # Try relative to module
            module_dir = Path(__file__).parent.parent
            file_path = module_dir / "al-quran.txt"
            
        if not file_path.exists():
            raise QuranFileError(f"Quran text file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as e:
            raise QuranFileError(f"Failed to read Quran file: {e}")
            
        if not content.strip():
            raise QuranFileError(f"Quran file is empty: {file_path}")

        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if chunk_by == "verse":
            # Each line is a verse
            chunks = [line for line in lines if len(line) >= min_length]
        
        elif chunk_by == "paragraph":
            # Group every N verses together
            n = STEERING_DEFAULTS.paragraph_verse_count
            chunks = []
            for i in range(0, len(lines), n):
                chunk = " ".join(lines[i:i+n])
                if len(chunk) >= min_length:
                    chunks.append(chunk)

        elif chunk_by == "surah":
            # Group by actual Surah boundaries
            chunks = []
            current_line = 0
            
            for verse_count in self.SURAH_VERSE_COUNTS:
                if current_line >= len(lines):
                    break
                    
                end_line = min(current_line + verse_count, len(lines))
                # Skip Bismillah if it's considered a separate line in some files
                # But here we assume strict line-per-verse mapping
                chunk = " ".join(lines[current_line:end_line])
                
                if len(chunk) >= min_length:
                    chunks.append(chunk)
                
                current_line = end_line
        else:
            chunks = lines

        logger.info(f"Loaded {len(chunks)} text chunks from Quran ({chunk_by} mode)")
        return chunks

    def create_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings [num_texts, embedding_dim]
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def create_quran_embeddings(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Literal["verse", "surah", "paragraph"] = "verse",
        batch_size: int = 32,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create embeddings for the entire Quran text.

        Args:
            file_path: Path to Quran text
            chunk_by: How to chunk the text
            batch_size: Batch size for encoding
            save_path: Optional path to save embeddings

        Returns:
            Dictionary with 'embeddings', 'texts', and 'mean_embedding'
        """
        texts = self.load_quran_text(file_path, chunk_by=chunk_by)
        embeddings = self.create_embeddings(texts, batch_size=batch_size)

        # Compute mean embedding (the "Quran vector")
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        result = {
            "embeddings": embeddings,
            "texts": texts,
            "mean_embedding": mean_embedding,
            "model_name": self.model_name,
            "chunk_by": chunk_by,
        }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path,
                embeddings=embeddings,
                mean_embedding=mean_embedding,
                model_name=np.array(self.model_name),
                chunk_by=np.array(chunk_by),
            )
            # Save texts separately (numpy doesn't handle variable-length strings well)
            with open(save_path.with_suffix(".texts.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            logger.info(f"Saved embeddings to {save_path}")

        return result

    def load_cached_embeddings(
        self,
        load_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load previously saved embeddings."""
        load_path = Path(load_path)
        data = np.load(load_path, allow_pickle=True)

        texts = []
        texts_path = load_path.with_suffix(".texts.txt")
        if texts_path.exists():
            with open(texts_path, "r", encoding="utf-8") as f:
                texts = f.read().split("\n")

        return {
            "embeddings": data["embeddings"],
            "mean_embedding": data["mean_embedding"],
            "texts": texts,
            "model_name": str(data.get("model_name", "unknown")),
            "chunk_by": str(data.get("chunk_by", "unknown")),
        }

    def get_semantic_clusters(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Cluster embeddings to find semantic themes.

        Args:
            embeddings: The embeddings array
            n_clusters: Number of clusters

        Returns:
            Dictionary with cluster centers and labels
        """
        from scipy.cluster.vq import kmeans2

        centers, labels = kmeans2(embeddings.astype(np.float64), n_clusters, minit="++")

        # Normalize cluster centers
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        return {
            "centers": centers.astype(np.float32),
            "labels": labels,
            "n_clusters": n_clusters,
        }
