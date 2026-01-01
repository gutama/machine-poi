"""
Quran Text Embeddings Module

Creates semantic embeddings from Quranic verses using models like:
- Qwen3-Embedding-8B (Alibaba-NLP/gte-Qwen2-7B-instruct or similar)
- BGE-M3 (BAAI/bge-m3)
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Literal
import numpy as np
import torch


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
        "bge-large-zh": "BAAI/bge-large-zh-v1.5",  # Good for Arabic/Chinese
        "multilingual-e5": "intfloat/multilingual-e5-large-instruct",
    }

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
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def load_model(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        model_path = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        print(f"Loading embedding model: {model_path}")

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

        print(f"Model loaded on {self.device}")

    def load_quran_text(
        self,
        file_path: Union[str, Path] = "al-quran.txt",
        chunk_by: Literal["verse", "surah", "paragraph"] = "verse",
        min_length: int = 10,
    ) -> List[str]:
        """
        Load and chunk the Quran text.

        Args:
            file_path: Path to the Quran text file
            chunk_by: How to split the text (verse, surah, paragraph)
            min_length: Minimum character length for a chunk

        Returns:
            List of text chunks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            # Try relative to module
            module_dir = Path(__file__).parent.parent
            file_path = module_dir / "al-quran.txt"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if chunk_by == "verse":
            # Each line is a verse
            chunks = [line for line in lines if len(line) >= min_length]
        elif chunk_by == "paragraph":
            # Group every N verses together
            n = 5
            chunks = []
            for i in range(0, len(lines), n):
                chunk = " ".join(lines[i:i+n])
                if len(chunk) >= min_length:
                    chunks.append(chunk)
        elif chunk_by == "surah":
            # Group by assumed surah boundaries (simplified)
            # In practice, you'd need surah markers
            n = 50
            chunks = []
            for i in range(0, len(lines), n):
                chunk = " ".join(lines[i:i+n])
                if len(chunk) >= min_length:
                    chunks.append(chunk)
        else:
            chunks = lines

        print(f"Loaded {len(chunks)} text chunks from Quran")
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
            print(f"Saved embeddings to {save_path}")

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
