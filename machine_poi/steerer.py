"""
Main Quran Steering Interface

High-level API for steering LLMs using Quran text embeddings.
Combines embedding extraction, steering vector creation, and LLM inference.
"""

import gc
import logging
import hashlib
import math
import threading
import warnings
from functools import wraps
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Literal, Any
from dataclasses import asdict, dataclass

from .controls import (
    calibration_texts,
    neutral_texts as neutral_control_texts,
    texts_sha256,
    unique_texts,
)
from .quran_embeddings import QuranEmbeddings, QuranFileError, resolve_corpus_path
from .llm_wrapper import SteeredLLM
from .steering_cache import CacheMismatchError, load_vectors, save_vectors
from .retrieval_context import quote_retrieval
from .knowledge_base import QuranKnowledgeBase
from .hybrid_knowledge_base import HybridQuranKnowledgeBase
from .graph_bridge import GraphBridgeGenerator
from .themes import (
    DOMAIN_BRIDGE_MAP,
    QURANIC_THEMES,
    embed_query,
    matching_keywords,
    theme_index,
)

from .config import (
    STEERING_DEFAULTS,
    MultiResolutionResults,
)


# Setup module logger
logger = logging.getLogger("machine_poi.steerer")


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

    # Dose as a target relative perturbation: at each steered layer the update
    # norm is this fraction of the layer's median token norm, measured on a
    # calibration set (QuranSteerer.calibrate_dose), times the layer
    # distribution scale. Negative values steer away. Add mode only. Set it to
    # None to use the raw coefficient below instead.
    dose_ratio: Optional[float] = 0.05

    # Raw coefficient, used only when dose_ratio is None. Its effect depends on
    # the model's activation scale, so it does not transfer between models.
    coefficient: float = 0.5

    # Which layers to steer (None = auto-select middle layers)
    target_layers: Optional[List[int]] = None

    # Injection mode: "add", "blend", "replace", "clamp"
    injection_mode: str = "add"  # Use "clamp" for higher coefficients

    # How to distribute steering across layers
    layer_distribution: Literal["uniform", "bell", "focused", "workspace"] = "bell"

    # For "focused" distribution, which relative layer (0-1)
    focus_layer: float = 0.5

    def validate(self) -> None:
        """Validate configuration values."""
        if self.dose_ratio is not None:
            if not math.isfinite(self.dose_ratio) or abs(self.dose_ratio) > 1:
                raise InvalidConfigError(
                    f"Dose ratio must be finite and in [-1, 1], got {self.dose_ratio}"
                )
            if self.injection_mode != "add":
                raise InvalidConfigError(
                    "Dose ratios apply to add mode; set dose_ratio=None and use "
                    f"coefficient for {self.injection_mode!r} mode"
                )

        if not 0.0 <= self.coefficient <= 2.0:
            raise InvalidConfigError(f"Coefficient must be between 0.0 and 2.0, got {self.coefficient}")
        
        if self.injection_mode == "blend" and self.coefficient > 1:
            raise InvalidConfigError("Blend coefficient must be in [0, 1]")

        if self.injection_mode not in ("add", "blend", "replace", "clamp"):
            raise InvalidConfigError(f"Invalid injection mode: {self.injection_mode}")
        
        if self.layer_distribution not in ("uniform", "bell", "focused", "workspace"):
            raise InvalidConfigError(f"Invalid layer distribution: {self.layer_distribution}")
        
        if not 0.0 <= self.focus_layer <= 1.0:
            raise InvalidConfigError(f"Focus layer must be between 0.0 and 1.0, got {self.focus_layer}")


def cited(item: Dict[str, Any]) -> str:
    """Format a retrieved item as a bullet with its surah:ayah reference."""
    ref = item.get("ref")
    return f"- [{ref}] {item['content']}" if ref else f"- {item['content']}"


def select_workspace_layers(num_layers: int) -> List[int]:
    """
    Select likely workspace-like intervention layers.

    Workspace-inspired steering emphasizes intermediate layers where model-native
    representations are expected to be more reusable for downstream computation,
    while avoiding very early parsing layers and late token-output layers.
    """
    if num_layers <= 0:
        return []
    start = int(np.floor(num_layers * 0.40))
    end = int(np.ceil(num_layers * 0.70))
    return list(range(max(0, start), min(num_layers, max(start + 1, end))))


def select_target_layers(
    num_layers: int,
    distribution: str,
    focus_layer: float = 0.5,
) -> List[int]:
    """Select default target layers for a steering distribution."""
    if num_layers <= 0:
        return []
    if distribution == "focused":
        center = int(round(focus_layer * (num_layers - 1)))
        return list(range(max(0, center - 2), min(num_layers, center + 3)))
    if distribution == "bell":
        start = num_layers // 3
        end = 2 * num_layers // 3
        return list(range(start, max(start + 1, end)))
    if distribution == "workspace":
        return select_workspace_layers(num_layers)
    return list(range(num_layers))


def layer_distribution_scale(layer_idx: int, num_layers: int, distribution: str) -> float:
    """Return the layer-specific steering scale for a distribution mode."""
    if num_layers <= 0:
        return 1.0
    if distribution == "bell":
        center = num_layers / 2
        return float(np.exp(-0.5 * ((layer_idx - center) / (num_layers / 4)) ** 2))
    if distribution == "workspace":
        center = num_layers * 0.55
        width = max(num_layers * 0.15, 1.0)
        return float(np.exp(-0.5 * ((layer_idx - center) / width) ** 2))
    return 1.0


def serialized(method):
    """Serialize high-level state changes; async retrieval stays outside scope."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._run_lock:
            return method(self, *args, **kwargs)
    return wrapped


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
        use_graph_kb: bool = False,  # Enable graph-based knowledge base
        llm_func: Optional[callable] = None,  # LLM function for LightRAG
        llm_revision: Optional[str] = None,
        trust_remote_code: bool = False,
        embedding_revision: Optional[str] = None,
    ):
        """
        Initialize the Quran steerer.

        Args:
            llm_model: Name/path of the LLM to steer
            embedding_model: Name/path of the embedding model
            quran_path: Path to Quran text file
            device: Device for computation
            llm_quantization: Optional quantization for LLM
            use_graph_kb: Whether to enable graph-based knowledge base
            llm_func: LLM function for LightRAG entity extraction

        Raises:
            FileNotFoundError: If quran_path doesn't exist
        """
        self._run_lock = threading.RLock()
        self.embedding_revision = embedding_revision
        self.llm_revision = llm_revision
        self.trust_remote_code = trust_remote_code
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model
        try:
            self.quran_path = resolve_corpus_path(quran_path)
        except QuranFileError as exc:
            raise FileNotFoundError(str(exc)) from exc

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

        # Graph-based knowledge base settings
        self.use_graph_kb = use_graph_kb
        self._llm_func = llm_func

        # Components (loaded lazily)
        self.embedder: Optional[QuranEmbeddings] = None
        self.llm: Optional[SteeredLLM] = None
        self.knowledge_base: Optional[QuranKnowledgeBase] = None
        self.hybrid_kb: Optional[HybridQuranKnowledgeBase] = None
        self.graph_bridge_generator: Optional[GraphBridgeGenerator] = None

        # Cached data
        self.quran_embeddings: Optional[Dict[str, Any]] = None
        self.steering_vectors: Optional[Dict[int, torch.Tensor]] = None
        self.config = SteeringConfig()
        # Median token norm per layer, measured by calibrate_dose
        self.dose_calibration: Optional[Dict[str, Any]] = None
        self.last_run_diagnostics = {}
        self.last_run_settings = {}
        
        logger.debug(f"Initialized QuranSteerer with model={llm_model}, device={self.device}")

    @serialized
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
                revision=self.embedding_revision,
            )
            self.embedder.load_model()

        if load_llm:
            logger.info("Loading LLM...")
            self.llm = SteeredLLM(
                model_name=self.llm_model_name,
                device=self.device,
                load_in_8bit=self.llm_quantization == "8bit",
                load_in_4bit=self.llm_quantization == "4bit",
                revision=self.llm_revision,
                trust_remote_code=self.trust_remote_code,
            )
            self.llm.load_model()
            self.dose_calibration = None  # norms belong to the previous model

    def initialize_knowledge_base(self, persist_dir: str = "quran_db") -> None:
        """Initialize the knowledge base, sharing the loaded embedder if any."""
        logger.info("Initializing Knowledge Base...")
        self.knowledge_base = QuranKnowledgeBase(
            persist_dir=persist_dir,
            embedding_model_name=self.embedding_model_name,
            device=self.device,
            embedder=self.embedder,
            quran_path=self.quran_path,
        )

    async def initialize_hybrid_knowledge_base(
        self,
        vector_persist_dir: str = "quran_db",
        graph_working_dir: str = "quran_lightrag",
    ) -> None:
        """
        Initialize hybrid knowledge base with graph support.
        
        Args:
            vector_persist_dir: Directory for ChromaDB vector storage
            graph_working_dir: Directory for LightRAG graph storage
        """
        logger.info("Initializing Hybrid Knowledge Base...")

        self.hybrid_kb = HybridQuranKnowledgeBase(
            vector_persist_dir=vector_persist_dir,
            graph_working_dir=graph_working_dir,
            embedding_model_name=self.embedding_model_name,
            device=self.device,
            llm_func=self._llm_func,
            embedder=self.embedder,
            quran_path=self.quran_path,
        )
        await self.hybrid_kb.initialize()

        # Set up graph-based bridge generator
        self.graph_bridge_generator = self.hybrid_kb._bridge_generator
        logger.info("Hybrid Knowledge Base initialized")

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
        self._ensure_embedder_loaded()
        return theme_index(self.embedder)

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
        
        query_embedding = embed_query(self.embedder, query)

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
        use_auto_bridge: bool = True,
        use_graph: bool = None,  # Enable graph-based bridges
    ) -> List[str]:
        """
        Generate domain bridge queries from user input.

        ENHANCED: Three-tier bridging approach:
        1. Static DOMAIN_BRIDGE_MAP heuristic (fast lookup)
        2. Graph-based entity traversal (relationship-aware)
        3. Embedding similarity fallback (when others fail)
        
        Args:
            query: User's input query
            max_bridges: Maximum number of bridges to return (default from config)
            use_auto_bridge: Whether to use embedding-based fallback
            use_graph: Whether to use graph-based bridging (auto-detected if None)
            
        Returns:
            List of bridge query strings
        """
        if use_graph is None:
            use_graph = self.use_graph_kb and self.graph_bridge_generator is not None

        if max_bridges is None:
            max_bridges = STEERING_DEFAULTS.max_domain_bridges
            
        bridges: List[str] = []

        # Tier 1: Try static DOMAIN_BRIDGE_MAP first (fast lookup)
        for keyword in matching_keywords(query, DOMAIN_BRIDGE_MAP):
            bridges.extend(DOMAIN_BRIDGE_MAP[keyword][:2])

        # Remove duplicates while preserving order
        seen: set = set()
        unique_bridges: List[str] = []
        for b in bridges:
            if b not in seen:
                seen.add(b)
                unique_bridges.append(b)

        bridge_queries = unique_bridges[:max_bridges]

        # Tier 2: Graph-based traversal (NEW)
        if not bridge_queries and use_graph and self.graph_bridge_generator:
            try:
                result = self.graph_bridge_generator.generate_bridges_sync(
                    query=query,
                    max_bridges=max_bridges,
                )
                bridge_queries = result.bridges
                if bridge_queries:
                    logger.info(f"Graph-based bridges: {bridge_queries}")
            except Exception as e:
                logger.warning(f"Graph bridge generation failed: {e}")

        # Tier 3: Fallback to embedding-based auto-bridge if no bridges found
        if not bridge_queries and use_auto_bridge and self.embedder is not None:
            bridge_queries = self._auto_bridge_via_embeddings(query, top_k=max_bridges)

        if bridge_queries:
            logger.info(f"Domain bridges: {bridge_queries}")

        return bridge_queries

    async def generate_with_graph(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        query_mode: str = "hybrid",
        use_dynamic_steering: bool = False,
        trusted_retrieval: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate with graph-enhanced MRA mode.

        Uses hybrid knowledge base for comprehensive retrieval
        combining vector similarity and graph traversal.
        
        Args:
            prompt: User prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            query_mode: Query mode ("vector", "graph", "hybrid", "auto")
            use_dynamic_steering: Whether to apply dynamic steering from results
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text response
        """
        if use_dynamic_steering and not trusted_retrieval:
            raise InvalidConfigError("Dynamic steering requires explicitly trusted retrieval")
        self._ensure_llm_loaded()

        if self.hybrid_kb is None:
            await self.initialize_hybrid_knowledge_base()

        # Query hybrid knowledge base
        result = await self.hybrid_kb.query(
            query=prompt,
            mode=query_mode,
            use_bridges=True,
        )

        return self._generate_graph_result(prompt, result, max_new_tokens,
                                           temperature, use_dynamic_steering, **kwargs)

    @serialized
    def _generate_graph_result(self, prompt, result, max_new_tokens,
                               temperature, use_dynamic_steering, **kwargs):
        self.last_run_diagnostics = {}
        self.last_run_settings = {}
        with self.llm.steering_session():
            # Apply dynamic steering from vector results
            if use_dynamic_steering and result.vector_results:
                dynamic_vectors = self.compute_dynamic_steering(result.vector_results)
                if dynamic_vectors:
                    self.apply_dynamic_steering(dynamic_vectors)

            # Build enhanced prompt with graph context
            context_parts = []

            # Graph answer provides high-level reasoning
            if result.graph_answer:
                context_parts.append(f"**Graph Analysis**:\n{result.graph_answer}")

            # Vector results provide specific verses
            if result.vector_results:
                verses = result.vector_results.get('verse', [])
                if verses:
                    verses_txt = "\n".join(cited(r) for r in verses[:3])
                    context_parts.append(f"**Relevant Verses**:\n{verses_txt}")

            # Bridges show the conceptual mapping
            if result.bridges:
                context_parts.append(f"**Thematic Bridges**: {', '.join(result.bridges)}")

            # Construct final prompt
            context = quote_retrieval("\n\n".join(context_parts), "hybrid_knowledge_base")

            final_prompt = (
                f"### Quranic Knowledge Context\n"
                f"{context}\n\n"
                f"### Task\n{prompt}\n\n"
                f"### Response\n"
            )

            output = self.llm.generate(
                prompt=final_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )

            self.last_run_diagnostics = self.llm.get_steering_diagnostics()
            self._record_settings(final_prompt, "graph")
            return output

    @serialized
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

        weights = torch.tensor([item["weight"] for item in texts_to_process], dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise InvalidConfigError("Retrieval weights must be finite, nonnegative and sum above zero")
        weights = weights / weights.sum()

        pooled = self._pooled_activations([item["text"] for item in texts_to_process])
        dynamic_vectors: Dict[int, torch.Tensor] = {
            layer_idx: self._unit(torch.sum(stacked * weights.unsqueeze(-1), dim=0))
            for layer_idx, stacked in pooled.items()
        }

        # Cleanup after processing
        self._cleanup_memory()
        
        return dynamic_vectors

    @serialized
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

        self.config.validate()
        if not math.isfinite(blend_ratio) or not 0 <= blend_ratio <= 1:
            raise InvalidConfigError("Dynamic blend ratio must be in [0, 1]")
        self._validate_vectors(dynamic_vectors)
        self._ensure_dose_calibration()
        # Clear existing steering
        self.llm.clear_steering()

        # Determine target layers
        target_layers = self.config.target_layers
        if target_layers is None:
            target_layers = select_target_layers(
                self.llm.num_layers,
                self.config.layer_distribution,
                self.config.focus_layer,
            )

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

            scaled_vector = blended_vec
            effective_coefficient = self._layer_coefficient(layer_idx, blended_vec)
            if self.config.injection_mode == "replace":
                scaled_vector = blended_vec * effective_coefficient

            self.llm.register_steering_hook(
                layer_idx=layer_idx,
                steering_vector=scaled_vector,
                coefficient=effective_coefficient,
                injection_mode=self.config.injection_mode,
            )

    def _pooled_activations(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        """Per-text mean activations at every layer, {layer: [num_texts, hidden]}."""
        return self.llm.pooled_layer_means(
            list(texts),
            batch_size=STEERING_DEFAULTS.activation_batch_size,
            exclude_special=STEERING_DEFAULTS.pool_exclude_special_tokens,
        )

    def _unit(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize a pooled direction and place it on the steering device."""
        return torch.nn.functional.normalize(vector, dim=-1).to(self.device)

    def _recipe_parameters(self, recipe: str, control: str = "ar") -> Dict[str, str]:
        """Validate a vector recipe; centered recipes name their control set."""
        if recipe == "centered":
            return {
                "control": control,
                "neutral_sha256": texts_sha256(neutral_control_texts(control)),
            }
        if recipe == "raw_mean":
            warnings.warn(
                "recipe='raw_mean' keeps the component every hidden state shares, so "
                "the direction is mostly generic model state rather than Quran "
                "content; use recipe='centered' unless reproducing older results",
                UserWarning,
                stacklevel=4,
            )
            return {}
        raise InvalidConfigError(f"Unknown recipe {recipe!r}; use 'centered' or 'raw_mean'")

    def _neutral_control_means(self, control: str = "ar") -> Dict[int, torch.Tensor]:
        """Mean pooled activation of a neutral control set per layer."""
        texts = neutral_control_texts(control)
        logger.info(f"Computing the neutral control mean from {len(texts)} {control!r} sentences...")
        pooled = self._pooled_activations(texts)
        return {layer_idx: stacked.mean(dim=0) for layer_idx, stacked in pooled.items()}

    def _direction(self, mean, control, layer_idx):
        """Unit steering direction: the mean, minus the control mean if centering."""
        if control is not None:
            mean = mean - control[layer_idx]
        return self._unit(mean)

    def _use_cached_vectors(self, cache_path, metadata) -> bool:
        """Apply matching cached vectors; log why a cache is not used."""
        try:
            vectors = load_vectors(cache_path, metadata)
            self.steering_vectors = {k: torch.tensor(v, device=self.device)
                                     for k, v in vectors.items()}
            self._apply_steering()
            return True
        except CacheMismatchError as exc:
            logger.warning("Recomputing steering vectors: %s", exc)
        except (ValueError, KeyError, OSError, EOFError) as exc:
            logger.warning("Cache rejected; recomputing: %s", type(exc).__name__)
        return False

    def _cache_metadata(self, method, **parameters):
        revision = self.llm_revision
        model = getattr(self.llm, "model", None)
        actual_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        if isinstance(actual_revision, str):
            revision = actual_revision
        # Format 2: verse sampling keeps every verse (format 1 dropped short ones).
        # Format 3: mean and persona vectors are centered on a neutral control by
        # default, and the vector recipe is recorded.
        return {"format": 3, "model": self.llm_model_name,
                "revision": revision or "unresolved",
                "corpus_sha256": hashlib.sha256(self.quran_path.read_bytes()).hexdigest(),
                "hidden_size": self.llm.hidden_size, "num_layers": self.llm.num_layers,
                "pooling": ("content_tokens" if STEERING_DEFAULTS.pool_exclude_special_tokens
                            else "all_tokens"),
                "method": method, "parameters": parameters}

    def _save_vectors(self, path, metadata):
        save_vectors(path, {k: v.detach().float().cpu().numpy()
                            for k, v in self.steering_vectors.items()}, metadata)

    def _validate_vectors(self, vectors):
        if not vectors:
            raise InvalidConfigError("Steering vectors cannot be empty")
        self._validate_layer_indices(list(vectors))
        for vector in vectors.values():
            if vector.shape != (self.llm.hidden_size,) or not torch.isfinite(vector).all():
                raise InvalidConfigError("Vector must have the model hidden dimension and finite values")

    @serialized
    def prepare_quran_steering(
        self,
        chunk_by: Literal["verse", "paragraph", "surah"] = "verse",
        cache_path: Optional[Union[str, Path]] = None,
        use_cached: bool = True,
        sample_size: Optional[int] = None,
        recipe: Literal["centered", "raw_mean"] = "centered",
        control: Literal["ar", "en"] = "ar",
        seed: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering vectors from Quran text using mean activations.

        The default ``centered`` recipe is contrastive activation addition:
        ``mean(Quran) - mean(neutral Arabic control)`` at each layer, so the
        direction keeps what distinguishes the verses rather than the large
        component every hidden state shares. ``raw_mean`` keeps the older,
        uncentered mean and warns.

        Args:
            chunk_by: How to chunk the Quran text
            cache_path: Path to cache the computed vectors
            use_cached: Whether to use cached vectors if available
            sample_size: Number of samples to use (default from config)
            recipe: "centered" (default) or "raw_mean"
            control: Neutral control language for centering: "ar" (default,
                language-matched) or "en" (the English set, for comparison)

        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        if self.embedder is None or self.llm is None:
            self.load_models()
            
        if sample_size is None:
            sample_size = STEERING_DEFAULTS.activation_sample_size

        if type(sample_size) is not int or sample_size < 1:
            raise InvalidConfigError("Sample size must be a positive integer")
        seed = STEERING_DEFAULTS.random_seed if seed is None else seed
        centering = self._recipe_parameters(recipe, control)
        metadata = self._cache_metadata("mean", recipe=recipe, **centering, chunk_by=chunk_by,
                                        sample_size=sample_size,
                                        seed=seed)
        if cache_path and use_cached and Path(cache_path).exists():
            if self._use_cached_vectors(cache_path, metadata):
                return self.steering_vectors

        # Load text
        texts = self.embedder.load_quran_text(self.quran_path, chunk_by=chunk_by)
        
        # Sample texts if too many
        if len(texts) > sample_size:
            logger.info(f"Sampling {sample_size} verses/chunks from {len(texts)} total...")
            rng = np.random.RandomState(seed)
            selected_texts = rng.choice(texts, size=sample_size, replace=False)
        else:
            selected_texts = texts

        logger.info("Computing mean activations from Quran text...")
        pooled = self._pooled_activations(list(selected_texts))
        control_means = self._neutral_control_means(control) if centering else None
        self.steering_vectors = {
            layer_idx: self._direction(stacked.mean(dim=0), control_means, layer_idx)
            for layer_idx, stacked in pooled.items()
        }

        if cache_path:
            self._save_vectors(cache_path, metadata)

        # Apply to LLM
        self._apply_steering()
        
        # Cleanup memory after heavy processing
        self._cleanup_memory()

        return self.steering_vectors

    @serialized
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

    @serialized
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

    @serialized
    def prepare_quran_persona(
        self,
        cache_dir: str = "vectors",
        verse_weight: float = 0.5,
        paragraph_weight: float = 0.35,
        surah_weight: float = 0.15,
        recipe: Literal["centered", "raw_mean"] = "centered",
    ) -> Dict[int, torch.Tensor]:
        """
        Create a "Quran Persona" by aggregating activations from all resolution levels.
        
        This computes a direction from verse, paragraph, and surah levels,
        then combines them with configurable weights to create a comprehensive
        steering profile. With the default ``centered`` recipe each level's
        direction is its mean minus the neutral Arabic control mean, as in
        :meth:`prepare_quran_steering`.

        Args:
            cache_dir: Directory to cache computed vectors
            verse_weight: Weight for verse-level activations
            paragraph_weight: Weight for paragraph-level activations
            surah_weight: Weight for surah-level activations
            recipe: "centered" (default) or "raw_mean"

        Returns:
            Dictionary mapping layer indices to combined steering vectors
        """
        if self.embedder is None or self.llm is None:
            self.load_models()
            
        cache_path = Path(cache_dir) / "quran_persona_multiresolution.npz"
        
        weights = (verse_weight, paragraph_weight, surah_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise InvalidConfigError("Persona weights must be finite, nonnegative and sum above zero")
        centering = self._recipe_parameters(recipe)
        metadata = self._cache_metadata("persona", recipe=recipe, **centering,
                                        weights=list(weights),
                                        sample_size=STEERING_DEFAULTS.persona_sample_size,
                                        seed=STEERING_DEFAULTS.random_seed)
        if cache_path.exists() and self._use_cached_vectors(cache_path, metadata):
            return self.steering_vectors

        # Normalize weights
        total_weight = verse_weight + paragraph_weight + surah_weight
        verse_weight /= total_weight
        paragraph_weight /= total_weight
        surah_weight /= total_weight
        
        logger.info("Computing multi-resolution Quran Persona...")
        
        # Collect activations from each resolution level
        resolution_activations: Dict[str, Dict[int, torch.Tensor]] = {}
        control = self._neutral_control_means() if centering else None
        
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
            
            pooled = self._pooled_activations(texts)
            resolution_activations[resolution] = {
                layer_idx: self._direction(stacked.mean(dim=0), control, layer_idx)
                for layer_idx, stacked in pooled.items()
            }
            
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
        
        self._save_vectors(cache_path, metadata)

        # Apply steering
        self._apply_steering()
        
        return self.steering_vectors

    @serialized
    def _apply_steering(self) -> None:
        """Apply current steering vectors to LLM."""
        if self.steering_vectors is None or self.llm is None:
            return
        
        # Validate before replacing a usable configuration.
        self.config.validate()
        self._validate_vectors(self.steering_vectors)

        target_layers = self.config.target_layers
        if target_layers is None:
            target_layers = select_target_layers(
                self.llm.num_layers,
                self.config.layer_distribution,
                self.config.focus_layer,
            )

        self._validate_layer_indices(target_layers)
        self._ensure_dose_calibration()
        self.llm.clear_steering()

        for layer_idx in target_layers:
            if layer_idx not in self.steering_vectors:
                continue

            vector = self.steering_vectors[layer_idx]
            scaled_vector = vector
            effective_coefficient = self._layer_coefficient(layer_idx, vector)
            if self.config.injection_mode == "replace":
                scaled_vector = vector * effective_coefficient

            self.llm.register_steering_hook(
                layer_idx=layer_idx,
                steering_vector=scaled_vector,
                coefficient=effective_coefficient,
                injection_mode=self.config.injection_mode,
            )

    @serialized
    def set_steering_strength(self, coefficient: float) -> None:
        """Switch to a raw coefficient without recomputing vectors.

        This turns off ratio dosing (``dose_ratio`` becomes None); use
        :meth:`set_dose_ratio` for a dose that transfers between models.
        """
        self._update_dose(dose_ratio=None, coefficient=coefficient)

    @serialized
    def set_dose_ratio(self, dose_ratio: float) -> None:
        """Set the target relative perturbation without recomputing vectors."""
        self._update_dose(dose_ratio=dose_ratio)

    def _update_dose(self, **changes) -> None:
        self._ensure_llm_loaded()
        previous = {name: getattr(self.config, name) for name in changes}
        for name, value in changes.items():
            setattr(self.config, name, value)
        try:
            self.config.validate()
        except InvalidConfigError:
            for name, value in previous.items():
                setattr(self.config, name, value)
            raise
        self._apply_steering()

    @serialized
    def calibrate_dose(self, texts: Optional[List[str]] = None) -> Dict[int, float]:
        """Measure the median token norm at every layer, the scale of dose ratios.

        Runs the unsteered model over neutral sentences, by default the English
        control set and ten Arabic control sentences. Applying a dose ratio
        calibrates automatically the first time; call this to use other texts.
        Special tokens are excluded as in pooling, and the median ignores
        attention-sink tokens with massive norms.

        Returns:
            Dict mapping layer index to its median token norm
        """
        self._ensure_llm_loaded()
        texts = unique_texts(calibration_texts() if texts is None else texts)
        if not texts:
            raise InvalidConfigError("Dose calibration needs at least one text")
        norms = self.llm.layer_token_norms(
            texts,
            batch_size=STEERING_DEFAULTS.activation_batch_size,
            exclude_special=STEERING_DEFAULTS.pool_exclude_special_tokens,
        )
        if not all(math.isfinite(norm) and norm > 0 for norm in norms.values()):
            raise SteeringError("Dose calibration found non-finite or zero token norms")
        self.dose_calibration = {
            "texts_sha256": texts_sha256(texts),
            "num_texts": len(texts),
            "layer_norms": norms,
        }
        logger.info(f"Calibrated dose on {len(texts)} texts")
        return norms

    def _ensure_dose_calibration(self) -> None:
        if self.config.dose_ratio is not None and self.dose_calibration is None:
            self.calibrate_dose()

    def _layer_coefficient(self, layer_idx: int, vector: torch.Tensor) -> float:
        """Hook coefficient for one layer: raw, or set to hit the dose ratio."""
        scale = layer_distribution_scale(
            layer_idx, self.llm.num_layers, self.config.layer_distribution
        )
        if self.config.dose_ratio is None:
            return scale * self.config.coefficient
        # Add mode moves every token by coefficient * |vector|.
        norm = self.dose_calibration["layer_norms"][layer_idx]
        vector_norm = float(vector.detach().float().norm().clamp_min(1e-8))
        return scale * self.config.dose_ratio * norm / vector_norm

    def _mra_context(self, prompt: str, use_domain_bridges: bool):
        """Retrieve multi-resolution context and build the MRA prompt.

        Returns the final prompt and the raw retrieval results. Retrieval does
        not depend on steering, so comparisons can reuse one prompt for both
        arms.
        """
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

        # 3. Construct MRA Prompt
        verses_txt = "\n".join(cited(r) for r in results['verse'])
        passages_txt = "\n".join(cited(r) for r in results['passage'])
        surahs_txt = "\n".join(cited(r) for r in results['surah'])

        verses_txt = quote_retrieval(verses_txt, "quran_db:verse")
        passages_txt = quote_retrieval(passages_txt, "quran_db:passage")
        surahs_txt = quote_retrieval(surahs_txt, "quran_db:surah")
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
        return final_prompt, results

    def _prepare_prompt(
        self,
        prompt: str,
        mra_mode: bool,
        use_domain_bridges: bool,
        use_dynamic_steering: bool,
        dynamic_blend_ratio: float,
    ) -> str:
        """Build the final prompt; call inside a steering session.

        Dynamic steering mutates hooks, which the enclosing session restores.
        """
        if not mra_mode:
            return prompt
        final_prompt, results = self._mra_context(prompt, use_domain_bridges)
        if use_dynamic_steering:
            dynamic_vectors = self.compute_dynamic_steering(results)
            if dynamic_vectors:
                self.apply_dynamic_steering(dynamic_vectors, blend_ratio=dynamic_blend_ratio)
                logger.info(f"Dynamic Steering Applied (blend={dynamic_blend_ratio})")
        return final_prompt

    def _record_settings(self, final_prompt: str, retrieval: str) -> None:
        """Keep the settings a run actually used, to report next to its outputs."""
        decoding = getattr(self.llm, "last_generation_settings", None)
        self.last_run_settings = {
            **(decoding if isinstance(decoding, dict) else {}),
            "retrieval": retrieval,
            "prompt_sha256": hashlib.sha256(final_prompt.encode()).hexdigest(),
            "steering": asdict(self.config),
            "layer_coefficients": {
                layer: hook.coefficient
                for layer, hook in getattr(self.llm, "hooks", {}).items()
                if getattr(hook, "enabled", False)
            },
        }
        if self.config.dose_ratio is not None and self.dose_calibration:
            self.last_run_settings["dose_calibration"] = {
                key: self.dose_calibration[key] for key in ("texts_sha256", "num_texts")
            }

    def _check_generation_options(self, use_dynamic_steering, trusted_retrieval, dynamic_blend_ratio):
        if use_dynamic_steering and not trusted_retrieval:
            raise InvalidConfigError("Dynamic steering requires explicitly trusted retrieval")
        self._ensure_llm_loaded()
        if dynamic_blend_ratio is None:
            dynamic_blend_ratio = STEERING_DEFAULTS.dynamic_blend_ratio
        return dynamic_blend_ratio

    @serialized
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        mra_mode: bool = False,
        use_domain_bridges: bool = True,
        use_dynamic_steering: bool = False,
        trusted_retrieval: bool = False,
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
        dynamic_blend_ratio = self._check_generation_options(
            use_dynamic_steering, trusted_retrieval, dynamic_blend_ratio
        )

        self.last_run_diagnostics = {}
        self.last_run_settings = {}
        with self.llm.steering_session():
            final_prompt = self._prepare_prompt(
                prompt, mra_mode, use_domain_bridges, use_dynamic_steering, dynamic_blend_ratio
            )
            output = self.llm.generate(
                prompt=final_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                reasoning_mode=reasoning_mode,
                **kwargs,
            )

            self.last_run_diagnostics = self.llm.get_steering_diagnostics()
            self._record_settings(final_prompt, "mra" if mra_mode else "none")
            return output

    @serialized
    def generate_unsteered(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """Generate text from the raw prompt without steering or retrieval."""
        self._ensure_llm_loaded()
        with self.llm.steering_disabled():
            return self.llm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    @serialized
    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        mra_mode: bool = False,
        use_domain_bridges: bool = True,
        use_dynamic_steering: bool = False,
        trusted_retrieval: bool = False,
        dynamic_blend_ratio: Optional[float] = None,
        reasoning_mode: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Compare steered vs unsteered outputs on identical inputs.

        Retrieval runs once and both arms receive the same final prompt and
        random seed, so differences come from steering rather than context or
        sampling noise. ``seed`` defaults to ``STEERING_DEFAULTS.random_seed``.

        Returns:
            Tuple of (steered_output, unsteered_output)
        """
        dynamic_blend_ratio = self._check_generation_options(
            use_dynamic_steering, trusted_retrieval, dynamic_blend_ratio
        )
        if seed is None:
            seed = STEERING_DEFAULTS.random_seed
        options = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reasoning_mode=reasoning_mode,
            seed=seed,
            **kwargs,
        )

        self.last_run_diagnostics = {}
        self.last_run_settings = {}
        with self.llm.steering_session():
            final_prompt = self._prepare_prompt(
                prompt, mra_mode, use_domain_bridges, use_dynamic_steering, dynamic_blend_ratio
            )
            steered = self.llm.generate(prompt=final_prompt, **options)
            # Read before the baseline pass overwrites captured activations.
            self.last_run_diagnostics = self.llm.get_steering_diagnostics()
            with self.llm.steering_disabled():
                baseline = self.llm.generate(prompt=final_prompt, **options)
            self._record_settings(final_prompt, "mra" if mra_mode else "none")
        return steered, baseline

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
        # Per-text pooled activations, {layer: [num_texts, hidden]}
        self.positive_activations: Optional[Dict[int, torch.Tensor]] = None
        self.negative_activations: Optional[Dict[int, torch.Tensor]] = None

    @serialized
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
        distinct = unique_texts(negative_texts)
        if not distinct:
            raise ValueError("negative_texts cannot be empty")
        if len(distinct) < len(negative_texts):
            # Repeats would silently reweight the negative mean.
            logger.info(f"Dropped {len(negative_texts) - len(distinct)} repeated negative texts")
        negative_texts = distinct
            
        if self.llm is None:
            self.load_models(load_embedder=False)
            
        logger.info(f"Computing contrastive vectors from {len(positive_texts)} positive and {len(negative_texts)} negative examples...")
        
        logger.info("Extracting positive and negative activations...")
        self.positive_activations = self._pooled_activations(positive_texts)
        self.negative_activations = self._pooled_activations(negative_texts)
        
        # Compute contrastive vectors: mean(positive) - mean(negative)
        logger.info("Computing contrastive steering vectors...")
        self.steering_vectors = {}
        
        for layer_idx in self.positive_activations.keys():
            if layer_idx not in self.negative_activations:
                continue
                
            pos_mean = self.positive_activations[layer_idx].mean(dim=0)
            neg_mean = self.negative_activations[layer_idx].mean(dim=0)
            self.steering_vectors[layer_idx] = self._unit(pos_mean - neg_mean)
        
        # Cache if requested
        if cache_path:
            cache_path = Path(cache_path)
            self._save_vectors(cache_path, self._cache_metadata("contrastive",
                positive_sha256=texts_sha256(positive_texts),
                negative_sha256=texts_sha256(negative_texts)))
            logger.info(f"Saved contrastive vectors to {cache_path}")
        
        # Apply steering
        self._apply_steering()
        
        # Cleanup
        self._cleanup_memory()
        
        logger.info(f"Contrastive steering prepared with {len(self.steering_vectors)} layers")
        return self.steering_vectors

    @serialized
    def prepare_quran_contrastive(
        self,
        neutral_texts: Optional[List[str]] = None,
        quran_sample_size: int = 50,
        neutral_sample_size: int = 50,
    ) -> Dict[int, torch.Tensor]:
        """
        Convenience method: use Quran as positive and generate neutral texts as negative.
        
        Args:
            neutral_texts: Neutral negatives; default is the Arabic control set
                (machine_poi.controls), so the contrast is not Arabic vs English
            quran_sample_size: Number of Quran verses to sample
            neutral_sample_size: Maximum number of distinct neutral texts to use
            
        Returns:
            Dictionary of contrastive steering vectors
        """
        if self.embedder is None:
            self.load_models()
        
        # Get Quran verses as positive examples
        quran_texts = self.embedder.load_quran_text(self.quran_path, chunk_by="verse")
        rng = np.random.RandomState(STEERING_DEFAULTS.random_seed)
        picks = rng.choice(len(quran_texts), size=min(quran_sample_size, len(quran_texts)), replace=False)
        positive_texts = [quran_texts[i] for i in picks]
        
        # Language-matched control: Arabic neutral prose, sampled without repeats
        if neutral_texts is None:
            neutral_texts = neutral_control_texts("ar")
            if len(neutral_texts) > neutral_sample_size:
                picks = rng.choice(len(neutral_texts), size=neutral_sample_size, replace=False)
                neutral_texts = [neutral_texts[i] for i in picks]

        return self.prepare_contrastive_steering(positive_texts, neutral_texts)
