"""
Knowledge Base Module using ChromaDB.

Manages multi-resolution indexing of the Quran:
1. Micro: Individual Verses
2. Meso: Passages (Thematic groups)
3. Macro: Surahs (Chapters)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import chromadb

from .quran_embeddings import QuranEmbeddings

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STEERING_DEFAULTS, MultiResolutionResults, RetrievalResult

# Setup logger
logger = logging.getLogger("machine_poi.knowledge_base")


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass


class EmptyCollectionError(KnowledgeBaseError):
    """Raised when querying an empty collection."""
    pass


class QuranKnowledgeBase:
    """
    Manages Quranic knowledge in ChromaDB with multi-resolution support.
    """

    def __init__(
        self,
        persist_dir: str = "quran_db",
        embedding_model_name: str = "paraphrase-minilm",
        device: Optional[str] = None,
    ):
        """
        Initialize the knowledge base.

        Args:
            persist_dir: Directory to store ChromaDB data
            embedding_model_name: Name of embedding model to use
            device: Computation device
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Initialize embedder
        self.embedder = QuranEmbeddings(model_name=embedding_model_name, device=device)
        self.embedder.load_model()

        # Define collections for different resolutions
        self.collections = {
            "verse": self.client.get_or_create_collection(
                name="quran_verses",
                metadata={"hnsw:space": "cosine"}
            ),
            "passage": self.client.get_or_create_collection(
                name="quran_passages",
                metadata={"hnsw:space": "cosine"}
            ),
            "surah": self.client.get_or_create_collection(
                name="quran_surahs",
                metadata={"hnsw:space": "cosine"}
            ),
        }

    def build_index(self, quran_path: Union[str, Path] = "al-quran.txt") -> None:
        """
        Build or rebuild the index from source text.
        """
        logger.info("Building Knowledge Base Index...")
        
        # 1. Index Verses (Micro)
        verses = self.embedder.load_quran_text(quran_path, chunk_by="verse")
        self._index_collection("verse", verses, batch_size=STEERING_DEFAULTS.verse_index_batch_size)

        # 2. Index Passages (Meso)
        passages = self.embedder.load_quran_text(quran_path, chunk_by="paragraph")
        self._index_collection("passage", passages, batch_size=STEERING_DEFAULTS.passage_index_batch_size)

        # 3. Index Surahs (Macro)
        surahs = self.embedder.load_quran_text(quran_path, chunk_by="surah")
        self._index_collection("surah", surahs, batch_size=STEERING_DEFAULTS.surah_index_batch_size)
        
        logger.info("Indexing complete!")

    def _index_collection(self, resolution: str, texts: List[str], batch_size: int) -> None:
        """Helper to index a specific resolution."""
        collection = self.collections[resolution]
        
        # Check if already populated
        if collection.count() > 0:
            logger.info(f"Collection {resolution} already has {collection.count()} items. Skipping.")
            return

        logger.info(f"Indexing {len(texts)} {resolution}s...")
        
        # Generate embeddings in batches
        embeddings = self.embedder.create_embeddings(texts, batch_size=batch_size)
        
        # Add to Chroma
        # We process addition in batches to avoid hitting message size limits
        total = len(texts)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_texts = texts[i:end]
            batch_embeddings = embeddings[i:end].tolist()
            batch_ids = [f"{resolution}_{j}" for j in range(i, end)]
            
            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=[{"resolution": resolution, "index": j} for j in range(i, end)]
            )

    def query_multiresolution(
        self,
        query_text: str,
        n_results: int = 3,
        include_embeddings: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Query all resolutions simultaneously.

        Args:
            query_text: The query string
            n_results: Number of results per resolution
            include_embeddings: Whether to include embeddings in results (for dynamic steering)

        Returns:
            Dict with 'verse', 'passage', 'surah' results.
        """
        # Embed query
        query_embedding = self.embedder.create_embeddings([query_text])[0].tolist()

        results = {}
        for res_name, collection in self.collections.items():
            # Adjust n_results for macro levels (fewer surahs needed)
            k = n_results
            if res_name == "surah":
                k = max(1, n_results // 3)

            # Include embeddings if requested
            include_fields = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include_fields.append("embeddings")

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=include_fields
            )

            # Formatter
            formatted = []
            if response["documents"]:
                docs = response["documents"][0]
                metas = response["metadatas"][0]
                dists = response["distances"][0]
                embeds = response.get("embeddings", [[None] * len(docs)])[0] if include_embeddings else [None] * len(docs)

                for doc, meta, dist, emb in zip(docs, metas, dists, embeds):
                    item = {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": 1.0 - dist  # Cosine distance to similarity
                    }
                    if include_embeddings and emb is not None:
                        item["embedding"] = emb
                    formatted.append(item)
            results[res_name] = formatted

        return results

    def query_with_bridges(
        self,
        original_query: str,
        bridge_queries: List[str],
        n_results: int = 3,
        include_embeddings: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Query using both original query and domain bridge queries.

        Combines results from the original query and bridge queries,
        de-duplicating and re-ranking by best score.

        Args:
            original_query: The user's original query
            bridge_queries: List of domain bridge queries
            n_results: Number of results per resolution
            include_embeddings: Whether to include embeddings

        Returns:
            Dict with 'verse', 'passage', 'surah' results (merged and ranked).
        """
        all_queries = [original_query] + bridge_queries

        # Collect all results
        merged_results = {"verse": {}, "passage": {}, "surah": {}}

        for query in all_queries:
            results = self.query_multiresolution(
                query,
                n_results=n_results,
                include_embeddings=include_embeddings
            )

            for res_name, items in results.items():
                for item in items:
                    doc_id = item["metadata"].get("index", item["content"][:50])
                    # Keep the highest scoring occurrence
                    if doc_id not in merged_results[res_name] or item["score"] > merged_results[res_name][doc_id]["score"]:
                        merged_results[res_name][doc_id] = item

        # Convert back to list and sort by score
        final_results = {}
        for res_name, items_dict in merged_results.items():
            sorted_items = sorted(items_dict.values(), key=lambda x: x["score"], reverse=True)
            final_results[res_name] = sorted_items[:n_results]

        return final_results

    def get_weighted_embedding(self, query_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Calculate a comprehensive strategy for steering based on retrieved results.
        This is a placeholder for more advanced logic.
        """
        # Not used for steering directly yet, but helpful for logic
        pass
