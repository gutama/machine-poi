"""
Graph-Based Domain Bridge Generator

Uses LightRAG knowledge graph to dynamically map user queries
to relevant Quranic themes through entity-relationship traversal.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from .lightrag_adapter import QuranLightRAG
    from .quran_embeddings import QuranEmbeddings

logger = logging.getLogger("machine_poi.graph_bridge")


@dataclass
class BridgeResult:
    """Result from graph-based bridge generation."""
    bridges: List[str]
    entities_found: List[str]
    relationships_traversed: List[Tuple[str, str, str]]
    confidence_scores: Dict[str, float]


class GraphBridgeGenerator:
    """
    Generates domain bridges using knowledge graph traversal.

    Three-tier approach:
    1. Direct entity matching (fast path)
    2. Graph neighbor traversal (relationship-based)
    3. Embedding similarity fallback (when graph fails)
    """

    # Pre-defined mappings from common terms to graph entity names
    # These help map user vocabulary to Quranic entity names
    TERM_TO_ENTITY: Dict[str, List[str]] = {
        # Technical terms → Quranic entities
        "debug": ["patience", "careful_examination", "wisdom"],
        "error": ["repentance", "forgiveness", "learning"],
        "optimize": ["ihsan", "excellence", "perfection"],
        "team": ["ummah", "brotherhood", "unity"],
        "leader": ["prophet", "responsibility", "justice"],
        "stress": ["sabr", "tawakkul", "peace"],
        "failure": ["perseverance", "hope", "resilience"],
        "success": ["gratitude", "humility", "shukr"],
        # Work/career terms
        "deadline": ["time_management", "responsibility", "trust"],
        "meeting": ["consultation", "shura", "wisdom"],
        "conflict": ["reconciliation", "patience", "justice"],
        "promotion": ["gratitude", "humility", "continued_effort"],
        # Emotional terms
        "anxiety": ["tawakkul", "dhikr", "peace"],
        "fear": ["courage", "trust", "hope"],
        "anger": ["patience", "forgiveness", "self_control"],
        "sadness": ["hope", "patience", "trust_in_allah"],
    }

    # Relationship types that indicate thematic relevance
    BRIDGE_RELATIONSHIPS: Set[str] = {
        "exemplifies",
        "teaches",
        "leads_to",
        "requires",
        "contrasts_with",
        "manifests_as",
        "is_aspect_of",
        "practiced_by",
    }

    def __init__(
        self,
        quran_lightrag: "QuranLightRAG",
        embedder: Optional["QuranEmbeddings"] = None,
        max_traversal_depth: int = 2,
        min_confidence: float = 0.3,
    ):
        """
        Initialize GraphBridgeGenerator.

        Args:
            quran_lightrag: QuranLightRAG instance for graph queries
            embedder: Optional embedder for similarity fallback
            max_traversal_depth: Maximum graph traversal depth
            min_confidence: Minimum confidence threshold for bridges
        """
        self.lightrag = quran_lightrag
        self.embedder = embedder
        self.max_depth = max_traversal_depth
        self.min_confidence = min_confidence

        # Cache for graph labels (entity names)
        self._entity_cache: Optional[List[str]] = None

    async def _get_entity_names(self) -> List[str]:
        """Get all entity names from the knowledge graph."""
        if self._entity_cache is None:
            self._entity_cache = await self.lightrag.get_graph_labels()
        return self._entity_cache

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract relevant terms from user query."""
        # Simple tokenization - can be enhanced with NLP
        query_lower = query.lower()
        terms = []

        # Check for known term mappings
        for term in self.TERM_TO_ENTITY:
            if term in query_lower:
                terms.append(term)

        return terms

    async def generate_bridges(
        self,
        query: str,
        max_bridges: int = 5,
        use_graph: bool = True,
        use_embedding_fallback: bool = True,
    ) -> BridgeResult:
        """
        Generate domain bridges for a query using the knowledge graph.

        Algorithm:
        1. Extract terms from query
        2. Map terms to graph entities
        3. Traverse relationships to find connected Quranic concepts
        4. Rank by relevance and return top bridges

        Args:
            query: User query string
            max_bridges: Maximum number of bridges to return
            use_graph: Whether to use graph traversal
            use_embedding_fallback: Whether to fall back to embeddings

        Returns:
            BridgeResult with bridges and metadata
        """
        bridges: List[str] = []
        entities_found: List[str] = []
        relationships: List[Tuple[str, str, str]] = []
        confidence: Dict[str, float] = {}

        # 1. Extract query terms
        terms = self._extract_query_terms(query)

        # 2. Map terms to seed entities
        seed_entities = []
        for term in terms:
            if term in self.TERM_TO_ENTITY:
                seed_entities.extend(self.TERM_TO_ENTITY[term])

        if not seed_entities and use_graph:
            # Try direct query to find relevant entities
            try:
                result = await self.lightrag.query(
                    query=f"What Quranic concepts relate to: {query}",
                    mode="global",
                    top_k=5,
                )
                # Parse entities from result (implementation depends on LightRAG output)
                # This is a simplified extraction
                if result.get("answer"):
                    entities_found.append(result["answer"][:100])
            except Exception as e:
                logger.warning(f"Graph query failed: {e}")

        # 3. Traverse graph from seed entities
        if use_graph and seed_entities:
            for entity in seed_entities[:3]:  # Limit seed entities
                try:
                    neighbors = await self.lightrag.get_entity_neighbors(
                        entity_name=entity,
                        max_depth=self.max_depth,
                        max_nodes=10,
                    )

                    if neighbors:
                        entities_found.append(entity)
                        # Extract bridge concepts from neighbors
                        # (Implementation depends on LightRAG's KG structure)

                except Exception as e:
                    logger.debug(f"No graph data for entity '{entity}': {e}")

        # 4. Fallback to embedding similarity
        if not bridges and use_embedding_fallback and self.embedder:
            # Import here to avoid circular imports
            from .steerer import QURANIC_THEMES

            query_emb = self.embedder.create_embeddings([query])[0]
            theme_embs = self.embedder.create_embeddings(QURANIC_THEMES)

            # Normalize
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            norms = np.linalg.norm(theme_embs, axis=1, keepdims=True)
            theme_embs = theme_embs / (norms + 1e-8)

            similarities = np.dot(theme_embs, query_emb)
            top_indices = np.argsort(similarities)[::-1][:max_bridges]

            for idx in top_indices:
                if similarities[idx] >= self.min_confidence:
                    bridges.append(QURANIC_THEMES[idx])
                    confidence[QURANIC_THEMES[idx]] = float(similarities[idx])

        # Add seed entity themes as bridges if graph traversal didn't yield results
        if not bridges:
            for entity in seed_entities[:max_bridges]:
                if entity not in bridges:
                    bridges.append(entity)
                    confidence[entity] = 0.5  # Default confidence

        return BridgeResult(
            bridges=bridges[:max_bridges],
            entities_found=entities_found,
            relationships_traversed=relationships,
            confidence_scores=confidence,
        )

    def generate_bridges_sync(
        self,
        query: str,
        max_bridges: int = 5,
    ) -> BridgeResult:
        """Synchronous wrapper for generate_bridges."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.generate_bridges(query, max_bridges)
        )
