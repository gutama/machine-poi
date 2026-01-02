"""
LightRAG Adapter for Quranic Knowledge Graph

Provides entity-relationship extraction and graph-based retrieval
for Quranic concepts, prophets, virtues, and commands.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from functools import partial
from dataclasses import dataclass

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger("machine_poi.lightrag_adapter")


@dataclass
class QuranicEntity:
    """Represents an extracted Quranic entity."""
    name: str
    entity_type: str  # prophet, virtue, command, concept, event
    arabic_name: Optional[str] = None
    description: Optional[str] = None
    source_verses: List[str] = None

    def __post_init__(self):
        if self.source_verses is None:
            self.source_verses = []


@dataclass
class QuranicRelationship:
    """Represents a relationship between Quranic entities."""
    source: str
    target: str
    relationship: str  # e.g., "father_of", "exemplifies", "leads_to"
    weight: float = 1.0
    source_verse: Optional[str] = None


class QuranLightRAG:
    """
    LightRAG wrapper specialized for Quranic knowledge graph construction.

    Features:
    - Custom entity types for Islamic concepts
    - Arabic-aware extraction prompts
    - Integration with existing QuranEmbeddings
    - Hybrid query modes (local, global, hybrid, mix)
    """

    # Quranic entity types for extraction
    ENTITY_TYPES = [
        "prophet",           # Prophets and messengers
        "angel",             # Angels (Jibreel, Mikael, etc.)
        "virtue",            # Moral virtues (sabr, shukr, taqwa)
        "command",           # Divine commands and prohibitions
        "concept",           # Theological concepts (tawhid, risalah)
        "event",             # Historical events (hijrah, badr)
        "place",             # Sacred places (Makkah, Madinah)
        "group",             # Groups of people (believers, hypocrites)
        "practice",          # Religious practices (salah, zakah)
        "scripture",         # Referenced scriptures (Torah, Injeel)
    ]

    def __init__(
        self,
        working_dir: str = "quran_lightrag",
        embedding_func: Optional[Callable] = None,
        embedding_dim: int = 384,
        llm_func: Optional[Callable] = None,
        llm_model_name: str = "gpt-4o-mini",
        chunk_token_size: int = 500,  # Smaller for verse-level granularity
        chunk_overlap: int = 50,
    ):
        """
        Initialize QuranLightRAG.

        Args:
            working_dir: Directory for LightRAG storage
            embedding_func: Custom embedding function (uses QuranEmbeddings if None)
            embedding_dim: Dimension of embeddings
            llm_func: Custom LLM function for entity extraction
            llm_model_name: Model name for LLM calls
            chunk_token_size: Token size per chunk
            chunk_overlap: Overlap between chunks
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.llm_model_name = llm_model_name
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap = chunk_overlap

        self._embedding_func = embedding_func
        self._llm_func = llm_func
        self._rag: Optional[LightRAG] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LightRAG with custom functions."""
        if self._initialized:
            return

        # Build embedding function wrapper
        if self._embedding_func is None:
            raise ValueError("embedding_func must be provided")

        embedding_wrapper = EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=self._embedding_func,
        )

        # Initialize LightRAG
        self._rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=self._llm_func,
            llm_model_name=self.llm_model_name,
            embedding_func=embedding_wrapper,
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap,
            # Quranic-specific settings
            addon_params={
                "language": "English",  # Or "Arabic" for Arabic corpus
                "entity_types": self.ENTITY_TYPES,
            },
            # Graph settings
            entity_extract_max_gleaning=2,
            top_k=10,
            max_graph_nodes=500,
        )

        await self._rag.initialize_storages()
        self._initialized = True
        logger.info(f"QuranLightRAG initialized at {self.working_dir}")

    async def finalize(self) -> None:
        """Cleanup LightRAG resources."""
        if self._rag:
            await self._rag.finalize_storages()
            self._initialized = False

    async def ingest_quran(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
    ) -> str:
        """
        Ingest Quranic text and build knowledge graph.

        Args:
            texts: List of Quranic text chunks (verses, passages)
            ids: Optional identifiers for each chunk

        Returns:
            Tracking ID for the ingestion
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Ingesting {len(texts)} Quranic texts...")

        # Join texts for batch processing
        combined = "\n\n".join(texts)

        track_id = await self._rag.ainsert(
            input=combined,
            ids=ids,
        )

        logger.info(f"Ingestion complete. Track ID: {track_id}")
        return track_id

    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the Quranic knowledge graph.

        Args:
            query: Query string
            mode: Query mode - "local", "global", "hybrid", or "mix"
                - local: Context-dependent, uses nearby graph structure
                - global: Leverages full knowledge structure
                - hybrid: Combines local and global
                - mix: Integrates graph and vector retrieval
            top_k: Number of results to return

        Returns:
            Dict with 'answer', 'entities', 'relationships', 'sources'
        """
        if not self._initialized:
            await self.initialize()

        param = QueryParam(
            mode=mode,
            top_k=top_k,
        )

        result = await self._rag.aquery(query, param=param)

        return {
            "answer": result,
            "mode": mode,
            "query": query,
        }

    async def get_entity_neighbors(
        self,
        entity_name: str,
        max_depth: int = 2,
        max_nodes: int = 20,
    ) -> Dict[str, Any]:
        """
        Get neighboring entities in the knowledge graph.

        Args:
            entity_name: Name of the entity to explore
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to return

        Returns:
            Dict with entity and its relationships
        """
        if not self._initialized:
            await self.initialize()

        kg = await self._rag.get_knowledge_graph(
            node_label=entity_name,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        return kg

    async def get_graph_labels(self) -> List[str]:
        """Get all entity labels in the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        return await self._rag.get_graph_labels()

    def query_sync(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Synchronous query wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.query(query, mode, top_k)
        )

    def get_entity_neighbors_sync(
        self,
        entity_name: str,
        max_depth: int = 2,
        max_nodes: int = 20,
    ) -> Dict[str, Any]:
        """Synchronous neighbor lookup wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_entity_neighbors(entity_name, max_depth, max_nodes)
        )
