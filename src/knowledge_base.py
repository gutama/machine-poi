"""
Knowledge Base Module using ChromaDB.

Manages multi-resolution indexing of the Quran:
1. Micro: Individual Verses
2. Meso: Passages (Thematic groups)
3. Macro: Surahs (Chapters)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm

from .quran_embeddings import QuranEmbeddings


class QuranKnowledgeBase:
    """
    Manages Quranic knowledge in ChromaDB with multi-resolution support.
    """

    def __init__(
        self,
        persist_dir: str = "quran_db",
        embedding_model_name: str = "bge-m3",
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

    def build_index(self, quran_path: Union[str, Path] = "al-quran.txt"):
        """
        Build or rebuild the index from source text.
        """
        print("Building Knowledge Base Index...")
        
        # 1. Index Verses (Micro)
        verses = self.embedder.load_quran_text(quran_path, chunk_by="verse")
        self._index_collection("verse", verses, batch_size=64)

        # 2. Index Passages (Meso)
        passages = self.embedder.load_quran_text(quran_path, chunk_by="paragraph")
        self._index_collection("passage", passages, batch_size=32)

        # 3. Index Surahs (Macro)
        surahs = self.embedder.load_quran_text(quran_path, chunk_by="surah")
        self._index_collection("surah", surahs, batch_size=8)
        
        print("Indexing complete!")

    def _index_collection(self, resolution: str, texts: List[str], batch_size: int):
        """Helper to index a specific resolution."""
        collection = self.collections[resolution]
        
        # Check if already populated
        if collection.count() > 0:
            print(f"Collection {resolution} already has {collection.count()} items. Skipping.")
            return

        print(f"Indexing {len(texts)} {resolution}s...")
        
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
    ) -> Dict[str, List[Dict]]:
        """
        Query all resolutions simultaneously.
        
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
            
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Formatter
            formatted = []
            if response["documents"]:
                docs = response["documents"][0]
                metas = response["metadatas"][0]
                dists = response["distances"][0]
                
                for doc, meta, dist in zip(docs, metas, dists):
                    formatted.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": 1.0 - dist  # Cosine distance to similarity
                    })
            results[res_name] = formatted
            
        return results

    def get_weighted_embedding(self, query_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Calculate a comprehensive strategy for steering based on retrieved results.
        This is a placeholder for more advanced logic.
        """
        # Not used for steering directly yet, but helpful for logic
        pass
