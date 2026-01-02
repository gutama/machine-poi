"""
Main Quran Steering Interface

High-level API for steering LLMs using Quran text embeddings.
Combines embedding extraction, steering vector creation, and LLM inference.
"""

import gc
import logging
import torch
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Literal, Any
from dataclasses import dataclass
from functools import lru_cache

from .quran_embeddings import QuranEmbeddings
from .steering_vectors import SteeringVectorExtractor, ContrastiveSteeringExtractor
from .llm_wrapper import SteeredLLM
from .knowledge_base import QuranKnowledgeBase

# Import config types and defaults
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STEERING_DEFAULTS,
    MultiResolutionResults,
    RetrievalResult,
)


# Setup module logger
logger = logging.getLogger("machine_poi.steerer")


# Domain bridge mappings: maps common concepts to Quranic themes
DOMAIN_BRIDGE_MAP: Dict[str, List[str]] = {
    # Technical/Programming domains
    "bug": ["correction", "improvement", "refinement", "fixing mistakes"],
    "debug": ["patience", "careful examination", "seeking truth"],
    "error": ["forgiveness", "learning from mistakes", "repentance"],
    "code": ["creation", "order", "structure", "wisdom"],
    "refactor": ["purification", "improvement", "renewal"],
    "optimize": ["excellence", "perfection", "ihsan"],
    "test": ["verification", "proof", "examination"],
    "deploy": ["trust in Allah", "tawakkul", "action after preparation"],

    # Teamwork/Social domains
    "team": ["unity", "brotherhood", "cooperation", "ummah"],
    "conflict": ["reconciliation", "peace-making", "patience"],
    "argue": ["respectful dialogue", "wisdom in speech", "reconciliation"],
    "collaborate": ["mutual help", "cooperation", "supporting one another"],
    "leadership": ["responsibility", "trust", "justice", "consultation"],
    "decision": ["consultation", "shura", "seeking guidance", "istikharah"],

    # Personal/Emotional domains
    "stress": ["patience", "sabr", "trust in Allah", "peace of heart"],
    "anxiety": ["remembrance of Allah", "tranquility", "tawakkul"],
    "failure": ["perseverance", "learning", "hope", "never despair"],
    "success": ["gratitude", "shukr", "humility", "continued effort"],
    "motivation": ["purpose", "intention", "seeking Allah's pleasure"],
    "fear": ["courage", "trust", "hope in Allah's mercy"],

    # Learning/Growth domains
    "learn": ["seeking knowledge", "wisdom", "reflection", "tadabbur"],
    "understand": ["contemplation", "insight", "divine guidance"],
    "teach": ["conveying truth", "patience", "wisdom", "example"],
    "growth": ["spiritual development", "self-improvement", "tarbiyah"],

    # General life domains
    "money": ["trust", "provision from Allah", "gratitude", "moderation"],
    "health": ["blessing", "patience in hardship", "gratitude"],
    "family": ["mercy", "compassion", "responsibility", "kindness to parents"],
    "time": ["value of time", "not wasting life", "preparation for hereafter"],
    "death": ["certainty", "preparation", "meeting Allah", "legacy"],
    "life": ["purpose", "test", "journey to Allah", "worship"],
}


# Curated Quranic themes for embedding-based auto-bridge generation
QURANIC_THEMES: List[str] = [
    # Core spiritual concepts
    "patience and perseverance (sabr)",
    "gratitude and thankfulness (shukr)",
    "trust and reliance on Allah (tawakkul)",
    "repentance and seeking forgiveness (tawbah)",
    "remembrance of Allah (dhikr)",
    "spiritual purification (tazkiyah)",
    "excellence in worship (ihsan)",
    "consciousness of Allah (taqwa)",
    
    # Moral virtues
    "honesty and truthfulness",
    "justice and fairness",
    "mercy and compassion",
    "humility and modesty",
    "generosity and charity",
    "kindness to parents and family",
    "fulfilling promises and trusts",
    "forgiving others",
    
    # Life guidance
    "dealing with hardship and trials",
    "hope and never despairing",
    "balance and moderation",
    "seeking knowledge and wisdom",
    "reflection and contemplation (tadabbur)",
    "taking responsibility",
    "preparing for the hereafter",
    "purpose and meaning of life",
    
    # Social relations
    "brotherhood and unity",
    "consultation and cooperation (shura)",
    "reconciliation and peace-making",
    "respectful dialogue",
    "supporting one another",
    "community (ummah)",
    
    # Work and action
    "striving with effort (jihad al-nafs)",
    "excellence in work",
    "fulfilling duties and obligations",
    "taking action after preparation",
    "persisting despite difficulties",
    "learning from mistakes",
    
    # Inner states
    "peace and tranquility of heart",
    "contentment and inner satisfaction",
    "overcoming fear and anxiety",
    "building confidence through faith",
    "finding strength in adversity",
]


class SteeringError(Exception):
    """Base exception for steering-related errors."""
    pass


class ModelNotLoadedError(SteeringError):
    """Raised when models are not loaded but required."""
    pass


class InvalidLayerError(SteeringError):
    """Raised when an invalid layer index is specified."""
    pass


class InvalidConfigError(SteeringError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class SteeringConfig:
    """Configuration for steering behavior."""

    # Steering strength (higher = stronger effect)
    coefficient: float = 0.5

    # Which layers to steer (None = auto-select middle layers)
    target_layers: Optional[List[int]] = None

    # Injection mode: "add", "blend", "replace", "clamp"
    injection_mode: str = "add"  # Use "clamp" for higher coefficients

    # How to distribute steering across layers
    layer_distribution: Literal["uniform", "bell", "focused"] = "bell"

    # For "focused" distribution, which relative layer (0-1)
    focus_layer: float = 0.5

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.coefficient <= 2.0:
            raise InvalidConfigError(f"Coefficient must be between 0.0 and 2.0, got {self.coefficient}")
        
        if self.injection_mode not in ("add", "blend", "replace", "clamp"):
            raise InvalidConfigError(f"Invalid injection mode: {self.injection_mode}")
        
        if self.layer_distribution not in ("uniform", "bell", "focused"):
            raise InvalidConfigError(f"Invalid layer distribution: {self.layer_distribution}")
        
        if not 0.0 <= self.focus_layer <= 1.0:
            raise InvalidConfigError(f"Focus layer must be between 0.0 and 1.0, got {self.focus_layer}")


class QuranSteerer:
    """
    Main interface for steering LLMs with Quran-derived embeddings.

    Example usage:
        steerer = QuranSteerer(llm_model="qwen3-0.6b", embedding_model="bge-m3")
        steerer.load_models()
        steerer.prepare_quran_steering()

        # Generate with Quran influence
        output = steerer.generate("Tell me about justice and mercy")

        # Compare with and without
        steered, baseline = steerer.compare("What is the meaning of life?")
    """

    def __init__(
        self,
        llm_model: str = "deepseek-r1-1.5b",
        embedding_model: str = "paraphrase-minilm",
        quran_path: Union[str, Path] = "al-quran.txt",
        device: Optional[str] = None,
        llm_quantization: Optional[str] = None,  # "4bit", "8bit", or None
    ):
        """
        Initialize the Quran steerer.

        Args:
            llm_model: Name/path of the LLM to steer
            embedding_model: Name/path of the embedding model
            quran_path: Path to Quran text file
            device: Device for computation
            llm_quantization: Optional quantization for LLM

        Raises:
            FileNotFoundError: If quran_path doesn't exist
        """
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model
        self.quran_path = Path(quran_path)
        
        # Validate quran path exists
        if not self.quran_path.exists():
            # Try relative to module
            module_dir = Path(__file__).parent.parent
            alt_path = module_dir / "al-quran.txt"
            if alt_path.exists():
                self.quran_path = alt_path
            else:
                raise FileNotFoundError(f"Quran text file not found: {quran_path}")

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.llm_quantization = llm_quantization

        # Components (loaded lazily)
        self.embedder: Optional[QuranEmbeddings] = None
        self.llm: Optional[SteeredLLM] = None
        self.vector_extractor: Optional[SteeringVectorExtractor] = None
        self.knowledge_base: Optional[QuranKnowledgeBase] = None

        # Cached data
        self.quran_embeddings: Optional[Dict[str, Any]] = None
        self.steering_vectors: Optional[Dict[int, torch.Tensor]] = None
        self._theme_embeddings: Optional[np.ndarray] = None  # For auto domain bridges
        self.config = SteeringConfig()
        
        logger.debug(f"Initialized QuranSteerer with model={llm_model}, device={self.device}")

    def load_models(self, load_llm: bool = True, load_embedder: bool = True) -> None:
        """
        Load the required models.

        Args:
            load_llm: Whether to load the LLM
            load_embedder: Whether to load the embedding model
        """
        if load_embedder:
            logger.info("Loading embedding model...")
            self.embedder = QuranEmbeddings(
                model_name=self.embedding_model_name,
                device=self.device,
            )
            self.embedder.load_model()

        if load_llm:
            logger.info("Loading LLM...")
            self.llm = SteeredLLM(
                model_name=self.llm_model_name,
                device=self.device,
                load_in_8bit=self.llm_quantization == "8bit",
                load_in_4bit=self.llm_quantization == "4bit",
            )
            self.llm.load_model()

    def initialize_knowledge_base(self, persist_dir: str = "quran_db") -> None:
        """Initialize the knowledge base."""
        logger.info("Initializing Knowledge Base...")
        self.knowledge_base = QuranKnowledgeBase(
            persist_dir=persist_dir,
            embedding_model_name=self.embedding_model_name,
            device=self.device,
        )

    def _ensure_llm_loaded(self) -> None:
        """Ensure LLM is loaded, raise error if not."""
        if self.llm is None:
            raise ModelNotLoadedError("LLM not loaded. Call load_models() first.")

    def _ensure_embedder_loaded(self) -> None:
        """Ensure embedder is loaded, raise error if not."""
        if self.embedder is None:
            raise ModelNotLoadedError("Embedding model not loaded. Call load_models() first.")

    def _validate_layer_indices(self, layer_indices: List[int]) -> None:
        """Validate that layer indices are within valid range."""
        self._ensure_llm_loaded()
        num_layers = self.llm.num_layers
        
        for idx in layer_indices:
            if not 0 <= idx < num_layers:
                raise InvalidLayerError(
                    f"Layer index {idx} is out of range. "
                    f"Model has {num_layers} layers (valid range: 0-{num_layers-1})"
                )

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory after heavy operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup completed")

    def _build_theme_index(self) -> np.ndarray:
        """
        Build embedding index for QURANIC_THEMES.
        
        Returns:
            Numpy array of shape (num_themes, embedding_dim) with normalized embeddings.
        """
        if self._theme_embeddings is not None:
            return self._theme_embeddings
            
        self._ensure_embedder_loaded()
        
        logger.info("Building theme embedding index for auto-bridge generation...")
        embeddings = self.embedder.create_embeddings(QURANIC_THEMES)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._theme_embeddings = embeddings / (norms + 1e-8)
        
        logger.info(f"Theme index built with {len(QURANIC_THEMES)} themes")
        return self._theme_embeddings

    def _auto_bridge_via_embeddings(
        self, 
        query: str, 
        top_k: int = 3,
        min_similarity: float = 0.3
    ) -> List[str]:
        """
        Generate domain bridges using embedding similarity when static lookup fails.
        
        Args:
            query: User's input query
            top_k: Maximum number of bridges to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of semantically similar Quranic themes
        """
        self._ensure_embedder_loaded()
        
        # Build theme index if not already built
        theme_embeddings = self._build_theme_index()
        
        # Embed the query
        query_embedding = self.embedder.create_embedding(query)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute cosine similarities
        similarities = np.dot(theme_embeddings, query_embedding)
        
        # Get top-k indices above threshold
        sorted_indices = np.argsort(similarities)[::-1]
        
        bridges = []
        for idx in sorted_indices[:top_k]:
            if similarities[idx] >= min_similarity:
                bridges.append(QURANIC_THEMES[idx])
        
        if bridges:
            logger.info(f"Auto-generated bridges via embeddings: {bridges}")
        
        return bridges

    def generate_domain_bridges(
        self, 
        query: str, 
        max_bridges: Optional[int] = None,
        use_auto_bridge: bool = True
    ) -> List[str]:
        """
        Generate domain bridge queries from user input.

        Maps concepts in the user's query to Quranic themes using:
        1. Static DOMAIN_BRIDGE_MAP heuristic (fast lookup)
        2. Embedding similarity fallback (when static fails)
        
        Args:
            query: User's input query
            max_bridges: Maximum number of bridges to return (default from config)
            use_auto_bridge: Whether to use embedding-based fallback
            
        Returns:
            List of bridge query strings
        """
        if max_bridges is None:
            max_bridges = STEERING_DEFAULTS.max_domain_bridges
            
        query_lower = query.lower()
        bridges: List[str] = []

        # 1. Try static DOMAIN_BRIDGE_MAP first (fast lookup)
        for keyword, themes in DOMAIN_BRIDGE_MAP.items():
            if keyword in query_lower:
                bridges.extend(themes[:2])

        # Remove duplicates while preserving order
        seen: set = set()
        unique_bridges: List[str] = []
        for b in bridges:
            if b not in seen:
                seen.add(b)
                unique_bridges.append(b)

        bridge_queries = unique_bridges[:max_bridges]

        # 2. Fallback to embedding-based auto-bridge if no static bridges found
        if not bridge_queries and use_auto_bridge and self.embedder is not None:
            bridge_queries = self._auto_bridge_via_embeddings(query, top_k=max_bridges)

        if bridge_queries:
            logger.info(f"Domain bridges: {bridge_queries}")

        return bridge_queries

    def compute_dynamic_steering(
        self,
        retrieved_results: MultiResolutionResults,
        resolution_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Compute steering vectors dynamically from retrieved content using ACTIVATIONS.
        
        Args:
            retrieved_results: Results from query_multiresolution
            resolution_weights: Optional weights for each resolution level
        
        Returns:
            Dictionary of steering vectors per layer, or None
        """
        self._ensure_llm_loaded()
        
        if resolution_weights is None:
            resolution_weights = STEERING_DEFAULTS.resolution_weights.copy()

        # Collect text content to process
        texts_to_process: List[Dict[str, Any]] = []
        
        for res_name, items in retrieved_results.items():
            res_weight = resolution_weights.get(res_name, 0.33)
            for item in items:
                text = item["content"]
                score = item.get("score", 0.5)
                texts_to_process.append({
                    "text": text,
                    "weight": res_weight * score
                })
        
        if not texts_to_process:
            logger.warning("No texts to process for dynamic steering")
            return None

        # Compute activations
        layer_activations: Dict[int, List[torch.Tensor]] = {}
        processed_weights: List[float] = []

        for item in texts_to_process:
            with torch.no_grad():
                activations = self.llm.extract_layer_activations(item["text"])
            
            for layer_idx, act in activations.items():
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                
                # Mean pooling over sequence
                mean_act = act.squeeze(0).mean(dim=0)
                layer_activations[layer_idx].append(mean_act)
            
            processed_weights.append(item["weight"])

        # Create weighted mean vector
        weights = torch.tensor(processed_weights, device=self.device)
        weights = weights / weights.sum()
        
        dynamic_vectors: Dict[int, torch.Tensor] = {}
        for layer_idx, act_list in layer_activations.items():
            # Stack: [num_items, hidden_dim]
            stacked = torch.stack(act_list)
            
            # Weighted average
            weighted_mean = torch.sum(stacked * weights.unsqueeze(-1), dim=0)
            
            # Normalize
            weighted_mean = torch.nn.functional.normalize(weighted_mean, dim=-1)
            dynamic_vectors[layer_idx] = weighted_mean

        # Cleanup after processing
        self._cleanup_memory()
        
        return dynamic_vectors

    def apply_dynamic_steering(
        self,
        dynamic_vectors: Dict[int, torch.Tensor],
        blend_ratio: Optional[float] = None,
    ) -> None:
        """
        Apply dynamic steering vectors, optionally blending with global steering.

        Args:
            dynamic_vectors: Steering vectors computed from retrieved content
            blend_ratio: How much to blend dynamic vs global (0=all global, 1=all dynamic)
        """
        if dynamic_vectors is None or self.llm is None:
            return
        
        if blend_ratio is None:
            blend_ratio = STEERING_DEFAULTS.dynamic_blend_ratio

        # Clear existing steering
        self.llm.clear_steering()

        # Determine target layers
        target_layers = self.config.target_layers
        if target_layers is None:
            num_layers = self.llm.num_layers
            if self.config.layer_distribution == "bell":
                start = num_layers // 3
                end = 2 * num_layers // 3
                target_layers = list(range(start, end))
            else:
                target_layers = list(range(num_layers))

        for layer_idx in target_layers:
            if layer_idx not in dynamic_vectors:
                continue

            dynamic_vec = dynamic_vectors[layer_idx]

            # Blend with global steering if available
            if self.steering_vectors and layer_idx in self.steering_vectors:
                global_vec = self.steering_vectors[layer_idx]
                blended_vec = (blend_ratio * dynamic_vec) + ((1 - blend_ratio) * global_vec)
            else:
                blended_vec = dynamic_vec

            # Apply layer-specific scaling
            if self.config.layer_distribution == "bell":
                center = self.llm.num_layers / 2
                scale = np.exp(-0.5 * ((layer_idx - center) / (self.llm.num_layers / 4)) ** 2)
            else:
                scale = 1.0

            scaled_vector = blended_vec * scale * self.config.coefficient

            self.llm.register_steering_hook(
                layer_idx=layer_idx,
                steering_vector=scaled_vector,
                coefficient=1.0,
                injection_mode=self.config.injection_mode,
            )

    def prepare_quran_steering(
        self,
        chunk_by: Literal["verse", "paragraph", "surah"] = "verse",
        cache_path: Optional[Union[str, Path]] = None,
        use_cached: bool = True,
        sample_size: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering vectors from Quran text using Mean Activation steering.
        
        Args:
            chunk_by: How to chunk the Quran text
            cache_path: Path to cache the computed vectors
            use_cached: Whether to use cached vectors if available
            sample_size: Number of samples to use (default from config)
            
        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        if self.embedder is None or self.llm is None:
            self.load_models()
            
        if sample_size is None:
            sample_size = STEERING_DEFAULTS.activation_sample_size

        # Try to load cached steering vectors directly
        if cache_path and use_cached:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached steering vectors...")
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    # Handle both npz formats
                    if 'steering_vectors' in data:
                        loaded_vecs = data['steering_vectors'].item()
                        self.steering_vectors = {
                            int(k): torch.tensor(v, device=self.device) 
                            for k, v in loaded_vecs.items()
                        }
                    else:
                        self.steering_vectors = {
                            int(k): torch.tensor(v, device=self.device) 
                            for k, v in data.items()
                        }
                    
                    # Validate dimensions match current LLM's hidden size
                    if self.steering_vectors:
                        sample_vec = next(iter(self.steering_vectors.values()))
                        expected_dim = self.llm.hidden_size
                        actual_dim = sample_vec.shape[-1]
                        if actual_dim != expected_dim:
                            logger.warning(
                                f"Cached vector dimension ({actual_dim}) doesn't match "
                                f"LLM hidden size ({expected_dim}). Invalidating cache..."
                            )
                            cache_path.unlink()  # Delete stale cache
                            self.steering_vectors = None
                        else:
                            self._apply_steering()
                            return self.steering_vectors
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # Load text
        texts = self.embedder.load_quran_text(self.quran_path, chunk_by=chunk_by)
        
        # Sample texts if too many
        if len(texts) > sample_size:
            logger.info(f"Sampling {sample_size} verses/chunks from {len(texts)} total...")
            rng = np.random.RandomState(STEERING_DEFAULTS.random_seed)
            selected_texts = rng.choice(texts, size=sample_size, replace=False)
        else:
            selected_texts = texts

        logger.info("Computing mean activations from Quran text...")
        
        layer_activations: Dict[int, List[torch.Tensor]] = {}
        
        for i, text in enumerate(selected_texts):
            if i % 10 == 0:
                logger.debug(f"Processing {i}/{len(selected_texts)}...")
                
            with torch.no_grad():
                activations = self.llm.extract_layer_activations(text)
                
            for layer_idx, act in activations.items():
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                
                # Mean pooling
                mean_act = act.squeeze(0).mean(dim=0)
                layer_activations[layer_idx].append(mean_act)

        # Compute global mean per layer
        self.steering_vectors = {}
        for layer_idx, act_list in layer_activations.items():
            stacked = torch.stack(act_list)
            global_mean = stacked.mean(dim=0)
            global_mean = torch.nn.functional.normalize(global_mean, dim=-1)
            self.steering_vectors[layer_idx] = global_mean

        # Save if requested
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            save_dict = {str(k): v.cpu().numpy() for k, v in self.steering_vectors.items()}
            np.savez(cache_path, **save_dict)
            logger.info(f"Saved steering vectors to {cache_path}")

        # Apply to LLM
        self._apply_steering()
        
        # Cleanup memory after heavy processing
        self._cleanup_memory()

        return self.steering_vectors

    def prepare_verse_steering(
        self,
        verse_indices: List[int],
        combine_method: Literal["mean", "max", "concat"] = "mean",
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering from specific verses (using activations on demand).
        
        Args:
            verse_indices: List of verse indices to use
            combine_method: How to combine verse activations
            
        Returns:
            Dictionary of steering vectors per layer
        """
        if self.embedder is None:
            self.load_models(load_llm=False, load_embedder=True)
        
        self._ensure_llm_loaded()

        texts = self.embedder.load_quran_text(self.quran_path, chunk_by="verse")
        
        # Validate indices
        for idx in verse_indices:
            if not 0 <= idx < len(texts):
                raise ValueError(f"Verse index {idx} out of range (0-{len(texts)-1})")
        
        selected_texts = [texts[i] for i in verse_indices]
        
        # Use dynamic steering logic to compute vectors
        fake_retrieval: MultiResolutionResults = {
            "verse": [{"content": t, "score": 1.0, "metadata": {}, "distance": 0.0} for t in selected_texts],
            "passage": [],
            "surah": []
        }
        
        vectors = self.compute_dynamic_steering(fake_retrieval)
        self.steering_vectors = vectors
        self._apply_steering()
        
        return vectors

    def prepare_thematic_steering(
        self,
        theme_query: str,
        top_k: int = 10,
    ) -> Dict[int, torch.Tensor]:
        """
        Steer using verses most similar to a theme query.
        
        Args:
            theme_query: Theme to search for (e.g., "mercy", "justice")
            top_k: Number of top verses to use
            
        Returns:
            Dictionary of steering vectors per layer
        """
        if self.embedder is None:
            self.load_models(load_llm=False)

        # Embed the query
        query_embedding = self.embedder.create_embeddings([theme_query])[0]
        
        # We need Quran embeddings for SIMILARITY SEARCH
        if self.quran_embeddings is None:
             self.quran_embeddings = self.embedder.create_quran_embeddings(
                file_path=self.quran_path,
                chunk_by="verse"
            )

        embeddings = self.quran_embeddings["embeddings"]
        similarities = embeddings @ query_embedding

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        logger.info(f"Top {top_k} verses for theme '{theme_query}':")
        for i, idx in enumerate(top_indices[:3]):
            logger.debug(f"  {i+1}. {self.quran_embeddings['texts'][idx][:50]}...")

        return self.prepare_verse_steering(list(top_indices))

    def prepare_quran_persona(
        self,
        cache_dir: str = "vectors",
        verse_weight: float = 0.5,
        paragraph_weight: float = 0.35,
        surah_weight: float = 0.15,
    ) -> Dict[int, torch.Tensor]:
        """
        Create a "Quran Persona" by aggregating activations from all resolution levels.
        
        This computes mean activations from verse, paragraph, and surah levels,
        then combines them with configurable weights to create a comprehensive
        steering profile.
        
        Args:
            cache_dir: Directory to cache computed vectors
            verse_weight: Weight for verse-level activations
            paragraph_weight: Weight for paragraph-level activations
            surah_weight: Weight for surah-level activations
            
        Returns:
            Dictionary mapping layer indices to combined steering vectors
        """
        if self.embedder is None or self.llm is None:
            self.load_models()
            
        cache_path = Path(cache_dir) / "quran_persona_multiresolution.npz"
        
        # Try to load cached
        if cache_path.exists():
            logger.info("Loading cached Quran Persona vectors...")
            try:
                data = np.load(cache_path, allow_pickle=True)
                self.steering_vectors = {
                    int(k): torch.tensor(v, device=self.device) 
                    for k, v in data.items()
                }
                self._apply_steering()
                return self.steering_vectors
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        
        # Normalize weights
        total_weight = verse_weight + paragraph_weight + surah_weight
        verse_weight /= total_weight
        paragraph_weight /= total_weight
        surah_weight /= total_weight
        
        logger.info("Computing multi-resolution Quran Persona...")
        
        # Collect activations from each resolution level
        resolution_activations: Dict[str, Dict[int, torch.Tensor]] = {}
        
        for resolution, weight, sample_size in [
            ("verse", verse_weight, STEERING_DEFAULTS.persona_sample_size),
            ("paragraph", paragraph_weight, STEERING_DEFAULTS.persona_sample_size // 2),
            ("surah", surah_weight, min(30, STEERING_DEFAULTS.persona_sample_size // 3)),
        ]:
            logger.info(f"Processing {resolution} level (weight={weight:.2f})...")
            
            texts = self.embedder.load_quran_text(self.quran_path, chunk_by=resolution)
            
            # Sample if needed
            if len(texts) > sample_size:
                rng = np.random.RandomState(STEERING_DEFAULTS.random_seed)
                texts = list(rng.choice(texts, size=sample_size, replace=False))
            
            layer_acts: Dict[int, List[torch.Tensor]] = {}
            
            for text in texts:
                with torch.no_grad():
                    activations = self.llm.extract_layer_activations(text)
                    
                for layer_idx, act in activations.items():
                    if layer_idx not in layer_acts:
                        layer_acts[layer_idx] = []
                    mean_act = act.squeeze(0).mean(dim=0)
                    layer_acts[layer_idx].append(mean_act)
            
            # Compute mean for this resolution
            resolution_activations[resolution] = {}
            for layer_idx, act_list in layer_acts.items():
                stacked = torch.stack(act_list)
                mean_vec = stacked.mean(dim=0)
                mean_vec = torch.nn.functional.normalize(mean_vec, dim=-1)
                resolution_activations[resolution][layer_idx] = mean_vec
            
            # Cleanup between resolutions
            self._cleanup_memory()
        
        # Combine all resolutions with weights
        logger.info("Combining multi-resolution activations...")
        self.steering_vectors = {}
        
        all_layers = set()
        for res_acts in resolution_activations.values():
            all_layers.update(res_acts.keys())
        
        weights_map = {"verse": verse_weight, "paragraph": paragraph_weight, "surah": surah_weight}
        
        for layer_idx in all_layers:
            combined = torch.zeros(self.llm.hidden_size, device=self.device)
            
            for resolution, acts in resolution_activations.items():
                if layer_idx in acts:
                    combined += weights_map[resolution] * acts[layer_idx]
            
            # Normalize the combined vector
            combined = torch.nn.functional.normalize(combined, dim=-1)
            self.steering_vectors[layer_idx] = combined
        
        # Cache the combined vectors
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        save_dict = {str(k): v.cpu().numpy() for k, v in self.steering_vectors.items()}
        np.savez(cache_path, **save_dict)
        logger.info(f"Saved Quran Persona vectors to {cache_path}")
        
        # Apply steering
        self._apply_steering()
        
        return self.steering_vectors

    def _apply_steering(self) -> None:
        """Apply current steering vectors to LLM."""
        if self.steering_vectors is None or self.llm is None:
            return
        
        # Validate config before applying
        self.config.validate()

        target_layers = self.config.target_layers
        if target_layers is None:
            num_layers = self.llm.num_layers
            if self.config.layer_distribution == "focused":
                center = int(self.config.focus_layer * num_layers)
                target_layers = list(range(max(0, center - 2), min(num_layers, center + 3)))
            elif self.config.layer_distribution == "bell":
                start = num_layers // 3
                end = 2 * num_layers // 3
                target_layers = list(range(start, end))
            else:
                target_layers = list(range(num_layers))

        for layer_idx in target_layers:
            if layer_idx not in self.steering_vectors:
                continue

            vector = self.steering_vectors[layer_idx]

            # Apply layer-specific scaling
            if self.config.layer_distribution == "bell":
                center = self.llm.num_layers / 2
                scale = np.exp(-0.5 * ((layer_idx - center) / (self.llm.num_layers / 4)) ** 2)
            else:
                scale = 1.0

            scaled_vector = vector * scale * self.config.coefficient

            self.llm.register_steering_hook(
                layer_idx=layer_idx,
                steering_vector=scaled_vector,
                coefficient=1.0, 
                injection_mode=self.config.injection_mode,
            )

    def set_steering_strength(self, coefficient: float) -> None:
        """Adjust steering strength without recomputing vectors."""
        self._ensure_llm_loaded()
        self.config.coefficient = coefficient
        self.llm.clear_steering()
        self._apply_steering()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        mra_mode: bool = False,
        use_domain_bridges: bool = True,
        use_dynamic_steering: bool = True,
        dynamic_blend_ratio: Optional[float] = None,
        reasoning_mode: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text with Quran-influenced steering.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            mra_mode: Enable Multi-Resolution Analysis
            use_domain_bridges: Enable domain bridging for MRA
            use_dynamic_steering: Enable dynamic steering for MRA
            dynamic_blend_ratio: Blend ratio for dynamic steering
            reasoning_mode: Enable model-specific reasoning mode
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        self._ensure_llm_loaded()
        
        if dynamic_blend_ratio is None:
            dynamic_blend_ratio = STEERING_DEFAULTS.dynamic_blend_ratio

        final_prompt = prompt

        if mra_mode:
            if self.knowledge_base is None:
                self.initialize_knowledge_base()

            # 1. Generate Domain Bridges
            bridge_queries: List[str] = []
            if use_domain_bridges:
                bridge_queries = self.generate_domain_bridges(prompt)

            # 2. Retrieve Multi-Resolution Context 
            if bridge_queries:
                results = self.knowledge_base.query_with_bridges(
                    original_query=prompt,
                    bridge_queries=bridge_queries,
                    n_results=3,
                    include_embeddings=False
                )
                logger.info(f"Domain Bridges Applied: {bridge_queries}")
            else:
                results = self.knowledge_base.query_multiresolution(
                    prompt,
                    n_results=3,
                    include_embeddings=False
                )

            # 3. Apply Dynamic Steering
            if use_dynamic_steering:
                dynamic_vectors = self.compute_dynamic_steering(results)
                if dynamic_vectors:
                    self.apply_dynamic_steering(dynamic_vectors, blend_ratio=dynamic_blend_ratio)
                    logger.info(f"Dynamic Steering Applied (blend={dynamic_blend_ratio})")

            # 4. Construct MRA Prompt
            verses_txt = "\n".join([f"- {r['content']}" for r in results['verse']])
            passages_txt = "\n".join([f"- {r['content']}" for r in results['passage']])
            surahs_txt = "\n".join([f"- {r['content']}" for r in results['surah']])

            bridges_section = ""
            if bridge_queries:
                bridges_section = f"**Domain Bridges**: {', '.join(bridge_queries)}\n\n"

            final_prompt = (
                f"### Quranic Multi-Resolution Context\n"
                f"{bridges_section}"
                f"**Micro (Verses):**\n{verses_txt}\n\n"
                f"**Meso (Passages):**\n{passages_txt}\n\n"
                f"**Macro (Surahs):**\n{surahs_txt}\n\n"
                f"### Task\n{prompt}\n\n"
                f"### Instruction\n"
                f"Perform a Multi-Resolution Analysis (MRA) and Multidomain Analogy:\n"
                f"1. **Micro Analysis**: How do the specific verses relate?\n"
                f"2. **Theme Analysis**: How do the broader passage themes apply?\n"
                f"3. **Multidomain Analogy**: Draw an analogy between these Quranic principles and the user's specific domain context.\n"
                f"4. **Synthesis**: Provide a clear answer based on this deep thinking.\n\n"
                f"### Response\n"
            )
            logger.info("MRA Context Injected")

        output = self.llm.generate(
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reasoning_mode=reasoning_mode,
            **kwargs,
        )

        if mra_mode and use_dynamic_steering:
            self.llm.clear_steering()
            self._apply_steering()

        return output

    def generate_unsteered(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """Generate text without steering."""
        self._ensure_llm_loaded()
        with self.llm.steering_disabled():
            return self.llm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[str, str]:
        """Compare steered vs unsteered outputs."""
        self._ensure_llm_loaded()
        return self.llm.compare_outputs(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def batch_compare(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> List[Tuple[str, str]]:
        """Compare outputs for multiple prompts."""
        results: List[Tuple[str, str]] = []
        for prompt in prompts:
            results.append(self.compare(prompt, max_new_tokens=max_new_tokens))
        return results

    def analyze_effect(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Analyze the steering effect across test prompts."""
        comparisons = self.batch_compare(test_prompts, max_new_tokens)

        analysis: Dict[str, Any] = {
            "prompts": test_prompts,
            "steered_outputs": [c[0] for c in comparisons],
            "unsteered_outputs": [c[1] for c in comparisons],
            "avg_length_steered": float(np.mean([len(c[0]) for c in comparisons])),
            "avg_length_unsteered": float(np.mean([len(c[1]) for c in comparisons])),
        }
        return analysis


class ContrastiveQuranSteerer(QuranSteerer):
    """
    Steerer using Contrastive Activation Addition (CAA).
    
    Uses paired positive/negative examples to determine steering direction.
    The steering vector is computed as: mean(positive_activations) - mean(negative_activations)
    
    Example usage:
        steerer = ContrastiveQuranSteerer(llm_model="qwen3-0.6b")
        steerer.load_models()
        
        # Use Quranic verses as positive, generic text as negative
        positive_texts = ["mercy and compassion...", "forgiveness and kindness..."]
        negative_texts = ["generic statement...", "neutral text..."]
        
        steerer.prepare_contrastive_steering(positive_texts, negative_texts)
        output = steerer.generate("Tell me about mercy")
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_extractor: Optional[ContrastiveSteeringExtractor] = None
        self.positive_activations: Optional[Dict[int, List[torch.Tensor]]] = None
        self.negative_activations: Optional[Dict[int, List[torch.Tensor]]] = None

    def prepare_contrastive_steering(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        cache_path: Optional[Union[str, Path]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering from contrastive pairs.
        
        Computes the contrastive steering vector as the difference between
        mean activations of positive and negative examples at each layer.
        
        Args:
            positive_texts: List of positive example texts (e.g., Quranic verses)
            negative_texts: List of negative example texts (e.g., neutral text)
            cache_path: Optional path to cache computed vectors
            
        Returns:
            Dictionary mapping layer indices to contrastive steering vectors
            
        Raises:
            ValueError: If text lists are empty
        """
        if not positive_texts:
            raise ValueError("positive_texts cannot be empty")
        if not negative_texts:
            raise ValueError("negative_texts cannot be empty")
            
        if self.llm is None:
            self.load_models(load_embedder=False)
            
        logger.info(f"Computing contrastive vectors from {len(positive_texts)} positive and {len(negative_texts)} negative examples...")
        
        # Initialize contrastive extractor
        if self.contrastive_extractor is None:
            self.contrastive_extractor = ContrastiveSteeringExtractor(
                target_dim=self.llm.hidden_size,
                device=self.device
            )
        
        # Extract positive activations
        logger.info("Extracting positive activations...")
        self.positive_activations = {}
        for i, text in enumerate(positive_texts):
            if i % 5 == 0:
                logger.debug(f"Processing positive {i}/{len(positive_texts)}...")
            
            with torch.no_grad():
                activations = self.llm.extract_layer_activations(text)
            
            for layer_idx, act in activations.items():
                if layer_idx not in self.positive_activations:
                    self.positive_activations[layer_idx] = []
                # Mean pool over sequence
                mean_act = act.squeeze(0).mean(dim=0)
                self.positive_activations[layer_idx].append(mean_act)
        
        # Extract negative activations
        logger.info("Extracting negative activations...")
        self.negative_activations = {}
        for i, text in enumerate(negative_texts):
            if i % 5 == 0:
                logger.debug(f"Processing negative {i}/{len(negative_texts)}...")
            
            with torch.no_grad():
                activations = self.llm.extract_layer_activations(text)
            
            for layer_idx, act in activations.items():
                if layer_idx not in self.negative_activations:
                    self.negative_activations[layer_idx] = []
                mean_act = act.squeeze(0).mean(dim=0)
                self.negative_activations[layer_idx].append(mean_act)
        
        # Compute contrastive vectors: mean(positive) - mean(negative)
        logger.info("Computing contrastive steering vectors...")
        self.steering_vectors = {}
        
        for layer_idx in self.positive_activations.keys():
            if layer_idx not in self.negative_activations:
                continue
                
            pos_stack = torch.stack(self.positive_activations[layer_idx])
            neg_stack = torch.stack(self.negative_activations[layer_idx])
            
            pos_mean = pos_stack.mean(dim=0)
            neg_mean = neg_stack.mean(dim=0)
            
            # Contrastive difference
            contrastive_vec = pos_mean - neg_mean
            
            # Normalize
            contrastive_vec = torch.nn.functional.normalize(contrastive_vec, dim=-1)
            
            self.steering_vectors[layer_idx] = contrastive_vec
            
            # Also compute for the extractor
            self.contrastive_extractor.compute_from_activations(
                positive_activations=self.positive_activations[layer_idx],
                negative_activations=self.negative_activations[layer_idx],
                layer_idx=layer_idx,
            )
        
        # Cache if requested
        if cache_path:
            cache_path = Path(cache_path)
            save_dict = {str(k): v.cpu().numpy() for k, v in self.steering_vectors.items()}
            np.savez(cache_path, **save_dict)
            logger.info(f"Saved contrastive vectors to {cache_path}")
        
        # Apply steering
        self._apply_steering()
        
        # Cleanup
        self._cleanup_memory()
        
        logger.info(f"Contrastive steering prepared with {len(self.steering_vectors)} layers")
        return self.steering_vectors

    def prepare_quran_contrastive(
        self,
        neutral_texts: Optional[List[str]] = None,
        quran_sample_size: int = 50,
        neutral_sample_size: int = 50,
    ) -> Dict[int, torch.Tensor]:
        """
        Convenience method: use Quran as positive and generate neutral texts as negative.
        
        Args:
            neutral_texts: Optional list of neutral texts. If None, generates simple prompts.
            quran_sample_size: Number of Quran verses to sample
            neutral_sample_size: Number of neutral texts to use
            
        Returns:
            Dictionary of contrastive steering vectors
        """
        if self.embedder is None:
            self.load_models()
        
        # Get Quran verses as positive examples
        quran_texts = self.embedder.load_quran_text(self.quran_path, chunk_by="verse")
        rng = np.random.RandomState(STEERING_DEFAULTS.random_seed)
        positive_texts = list(rng.choice(quran_texts, size=min(quran_sample_size, len(quran_texts)), replace=False))
        
        # Generate neutral texts if not provided
        if neutral_texts is None:
            neutral_prompts = [
                "The weather today is mild.",
                "Numbers are mathematical concepts.",
                "Water is composed of hydrogen and oxygen.",
                "Computers process information.",
                "Colors are perceived differently.",
                "Sound travels through air.",
                "Plants need sunlight to grow.",
                "Time passes continuously.",
                "Objects have mass and volume.",
                "Languages have grammar rules.",
            ]
            # Repeat to get enough samples
            neutral_texts = (neutral_prompts * (neutral_sample_size // len(neutral_prompts) + 1))[:neutral_sample_size]
        
        return self.prepare_contrastive_steering(positive_texts, neutral_texts)
