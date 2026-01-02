"""
Hybrid Knowledge Base

Combines ChromaDB vector retrieval with LightRAG knowledge graph
for comprehensive Quranic knowledge access.
"""

import logging
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

if TYPE_CHECKING:
    from .knowledge_base import QuranKnowledgeBase
    from .lightrag_adapter import QuranLightRAG
    from .graph_bridge import GraphBridgeGenerator

logger = logging.getLogger("machine_poi.hybrid_kb")


@dataclass
class HybridQueryResult:
    """Result from hybrid knowledge base query."""
    # Vector-based results (existing)
    vector_results: Dict[str, List[Dict]]

    # Graph-based results (new)
    graph_answer: Optional[str]
    graph_entities: List[str]
    graph_relationships: List[Dict]

    # Generated bridges
    bridges: List[str]
    bridge_confidence: Dict[str, float]

    # Fusion metadata
    query_mode: str
    fusion_strategy: str


class HybridQuranKnowledgeBase:
    """
    Hybrid knowledge base combining vector and graph retrieval.

    Query Modes:
    - "vector": ChromaDB only (fast, semantic similarity)
    - "graph": LightRAG only (relationship-aware)
    - "hybrid": Both with fusion (comprehensive)
    - "auto": Automatically select based on query type
    """

    def __init__(
        self,
        vector_persist_dir: str = "quran_db",
        graph_working_dir: str = "quran_lightrag",
        embedding_model_name: str = "paraphrase-minilm",
        device: Optional[str] = None,
        llm_func: Optional[callable] = None,
        llm_model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize hybrid knowledge base.

        Args:
            vector_persist_dir: Directory for ChromaDB
            graph_working_dir: Directory for LightRAG
            embedding_model_name: Embedding model name
            device: Computation device
            llm_func: LLM function for graph extraction
            llm_model_name: LLM model name
        """
        self.vector_persist_dir = vector_persist_dir
        self.graph_working_dir = graph_working_dir
        self.embedding_model_name = embedding_model_name
        self.device = device

        # Components (initialized lazily)
        self._vector_kb: Optional["QuranKnowledgeBase"] = None
        self._graph_kb: Optional["QuranLightRAG"] = None
        self._bridge_generator: Optional["GraphBridgeGenerator"] = None

        self._llm_func = llm_func
        self._llm_model_name = llm_model_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize both knowledge bases."""
        if self._initialized:
            return

        # Lazy imports to avoid circular dependencies
        from .knowledge_base import QuranKnowledgeBase
        from .lightrag_adapter import QuranLightRAG
        from .graph_bridge import GraphBridgeGenerator

        # Initialize vector KB (existing)
        self._vector_kb = QuranKnowledgeBase(
            persist_dir=self.vector_persist_dir,
            embedding_model_name=self.embedding_model_name,
            device=self.device,
        )

        # Create embedding function for LightRAG
        async def embedding_func(texts: List[str]) -> List[List[float]]:
            embeddings = self._vector_kb.embedder.create_embeddings(texts)
            return embeddings.tolist()

        # Initialize graph KB (new)
        embedding_dim = self._vector_kb.embedder.model.get_sentence_embedding_dimension()
        self._graph_kb = QuranLightRAG(
            working_dir=self.graph_working_dir,
            embedding_func=embedding_func,
            embedding_dim=embedding_dim,
            llm_func=self._llm_func,
            llm_model_name=self._llm_model_name,
        )
        await self._graph_kb.initialize()

        # Initialize bridge generator
        self._bridge_generator = GraphBridgeGenerator(
            quran_lightrag=self._graph_kb,
            embedder=self._vector_kb.embedder,
        )

        self._initialized = True
        logger.info("Hybrid Knowledge Base initialized")

    async def finalize(self) -> None:
        """Cleanup resources."""
        if self._graph_kb:
            await self._graph_kb.finalize()
        self._initialized = False

    async def build_index(
        self,
        quran_path: Union[str, Path] = "al-quran.txt",
        build_vector: bool = True,
        build_graph: bool = True,
    ) -> None:
        """
        Build both vector and graph indices.

        Args:
            quran_path: Path to Quran text file
            build_vector: Whether to build ChromaDB index
            build_graph: Whether to build LightRAG graph
        """
        if not self._initialized:
            await self.initialize()

        if build_vector:
            logger.info("Building vector index...")
            self._vector_kb.build_index(quran_path)

        if build_graph:
            logger.info("Building knowledge graph...")
            # Load texts for graph ingestion
            texts = self._vector_kb.embedder.load_quran_text(
                quran_path,
                chunk_by="verse"
            )
            await self._graph_kb.ingest_quran(texts)

    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        n_vector_results: int = 3,
        n_graph_results: int = 5,
        use_bridges: bool = True,
        max_bridges: int = 3,
        fusion_strategy: str = "interleave",
    ) -> HybridQueryResult:
        """
        Query the hybrid knowledge base.

        Args:
            query: Query string
            mode: Query mode ("vector", "graph", "hybrid", "auto")
            n_vector_results: Number of vector results per resolution
            n_graph_results: Number of graph results
            use_bridges: Whether to use domain bridges
            max_bridges: Maximum bridges to generate
            fusion_strategy: How to combine results ("interleave", "graph_first", "vector_first")

        Returns:
            HybridQueryResult with combined results
        """
        if not self._initialized:
            await self.initialize()

        # Auto mode: use graph for conceptual queries, vector for specific lookups
        if mode == "auto":
            # Simple heuristic: use graph if query seems conceptual
            conceptual_indicators = ["why", "how", "meaning", "purpose", "teach", "learn"]
            if any(ind in query.lower() for ind in conceptual_indicators):
                mode = "hybrid"
            else:
                mode = "vector"

        vector_results = {}
        graph_answer = None
        graph_entities = []
        graph_relationships = []
        bridges = []
        bridge_confidence = {}

        # Generate bridges
        if use_bridges:
            bridge_result = await self._bridge_generator.generate_bridges(
                query=query,
                max_bridges=max_bridges,
            )
            bridges = bridge_result.bridges
            bridge_confidence = bridge_result.confidence_scores

        # Vector retrieval
        if mode in ("vector", "hybrid"):
            if bridges:
                vector_results = self._vector_kb.query_with_bridges(
                    original_query=query,
                    bridge_queries=bridges,
                    n_results=n_vector_results,
                )
            else:
                vector_results = self._vector_kb.query_multiresolution(
                    query_text=query,
                    n_results=n_vector_results,
                )

        # Graph retrieval
        if mode in ("graph", "hybrid"):
            try:
                graph_result = await self._graph_kb.query(
                    query=query,
                    mode="hybrid",
                    top_k=n_graph_results,
                )
                graph_answer = graph_result.get("answer")
                # Extract entities and relationships from graph result
                # (Implementation depends on LightRAG's output format)
            except Exception as e:
                logger.warning(f"Graph query failed: {e}")

        return HybridQueryResult(
            vector_results=vector_results,
            graph_answer=graph_answer,
            graph_entities=graph_entities,
            graph_relationships=graph_relationships,
            bridges=bridges,
            bridge_confidence=bridge_confidence,
            query_mode=mode,
            fusion_strategy=fusion_strategy,
        )

    def query_sync(
        self,
        query: str,
        mode: str = "hybrid",
        **kwargs,
    ) -> HybridQueryResult:
        """Synchronous query wrapper."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.query(query, mode, **kwargs)
        )
