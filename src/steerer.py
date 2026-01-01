"""
Main Quran Steering Interface

High-level API for steering LLMs using Quran text embeddings.
Combines embedding extraction, steering vector creation, and LLM inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Literal
from dataclasses import dataclass

from .quran_embeddings import QuranEmbeddings
from .steering_vectors import SteeringVectorExtractor, ContrastiveSteeringExtractor
from .llm_wrapper import SteeredLLM
from .knowledge_base import QuranKnowledgeBase


# Domain bridge mappings: maps common concepts to Quranic themes
DOMAIN_BRIDGE_MAP = {
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


@dataclass
class SteeringConfig:
    """Configuration for steering behavior."""

    # Steering strength (higher = stronger effect)
    coefficient: float = 0.5

    # Which layers to steer (None = auto-select middle layers)
    target_layers: Optional[List[int]] = None

    # Injection mode: "add", "blend", "replace"
    injection_mode: str = "add"

    # How to distribute steering across layers
    layer_distribution: Literal["uniform", "bell", "focused"] = "bell"

    # For "focused" distribution, which relative layer (0-1)
    focus_layer: float = 0.5


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
        llm_model: str = "qwen3-0.6b",
        embedding_model: str = "bge-m3",
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
        """
        self.llm_model_name = llm_model
        self.embedding_model_name = embedding_model
        self.quran_path = Path(quran_path)

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
        self.quran_embeddings: Optional[Dict] = None
        self.steering_vectors: Optional[Dict[int, torch.Tensor]] = None
        self.config = SteeringConfig()

    def load_models(self, load_llm: bool = True, load_embedder: bool = True):
        """
        Load the required models.

        Args:
            load_llm: Whether to load the LLM
            load_embedder: Whether to load the embedding model
        """
        if load_embedder:
            print("Loading embedding model...")
            self.embedder = QuranEmbeddings(
                model_name=self.embedding_model_name,
                device=self.device,
            )
            self.embedder.load_model()

        if load_llm:
            print("Loading LLM...")
            self.llm = SteeredLLM(
                model_name=self.llm_model_name,
                device=self.device,
                load_in_8bit=self.llm_quantization == "8bit",
                load_in_4bit=self.llm_quantization == "4bit",
            )
            self.llm.load_model()

            # Initialize vector extractor with correct dimensions
            # Note: embedding dim may differ from LLM hidden dim
            # We'll handle this when we have both models loaded

    def initialize_knowledge_base(self, persist_dir: str = "quran_db"):
        """Initialize the knowledge base."""
        print("Initializing Knowledge Base...")
        self.knowledge_base = QuranKnowledgeBase(
            persist_dir=persist_dir,
            embedding_model_name=self.embedding_model_name,
            device=self.device,
        )

    def generate_domain_bridges(self, query: str, max_bridges: int = 3) -> List[str]:
        """
        Generate domain bridge queries from user input.

        Maps concepts in the user's query to Quranic themes using
        the DOMAIN_BRIDGE_MAP heuristic.

        Args:
            query: User's original query
            max_bridges: Maximum number of bridge queries to generate

        Returns:
            List of domain bridge query strings
        """
        query_lower = query.lower()
        bridges = []

        # Find matching keywords in the query
        for keyword, themes in DOMAIN_BRIDGE_MAP.items():
            if keyword in query_lower:
                # Add the first theme as a bridge
                bridges.extend(themes[:2])

        # Remove duplicates while preserving order
        seen = set()
        unique_bridges = []
        for b in bridges:
            if b not in seen:
                seen.add(b)
                unique_bridges.append(b)

        # Limit to max_bridges
        bridge_queries = unique_bridges[:max_bridges]

        if bridge_queries:
            print(f"Domain bridges: {bridge_queries}")

        return bridge_queries

    def generate_domain_bridges_llm(self, query: str, max_bridges: int = 3) -> List[str]:
        """
        Generate domain bridges using the LLM itself.

        This is more sophisticated but slower than the heuristic approach.

        Args:
            query: User's original query
            max_bridges: Maximum number of bridges to generate

        Returns:
            List of domain bridge query strings
        """
        if self.llm is None:
            return self.generate_domain_bridges(query, max_bridges)

        bridge_prompt = (
            f"Given the following user query, identify {max_bridges} Quranic/Islamic themes "
            f"that could provide relevant wisdom. Output ONLY the themes as a comma-separated list.\n\n"
            f"Query: {query}\n\n"
            f"Themes:"
        )

        # Generate with steering disabled for neutral bridging
        with self.llm.steering_disabled():
            response = self.llm.generate(bridge_prompt, max_new_tokens=50, temperature=0.3)

        # Parse the response
        themes = [t.strip() for t in response.split(",")]
        themes = [t for t in themes if t and len(t) < 50]  # Filter invalid

        return themes[:max_bridges]

    def compute_dynamic_steering(
        self,
        retrieved_results: Dict[str, List[Dict]],
        resolution_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Compute steering vectors dynamically from retrieved embeddings.

        Args:
            retrieved_results: Results from query_multiresolution with embeddings
            resolution_weights: Optional weights for each resolution level
                               Default: {"verse": 0.5, "passage": 0.35, "surah": 0.15}

        Returns:
            Dictionary of steering vectors per layer, or None if no embeddings
        """
        if resolution_weights is None:
            resolution_weights = {"verse": 0.5, "passage": 0.35, "surah": 0.15}

        # Collect all embeddings with their weights
        all_embeddings = []
        all_weights = []

        for res_name, items in retrieved_results.items():
            res_weight = resolution_weights.get(res_name, 0.33)
            for item in items:
                if "embedding" in item and item["embedding"] is not None:
                    emb = np.array(item["embedding"])
                    score = item.get("score", 0.5)
                    # Weight = resolution weight * similarity score
                    weight = res_weight * score
                    all_embeddings.append(emb)
                    all_weights.append(weight)

        if not all_embeddings:
            return None

        # Compute weighted average embedding
        all_embeddings = np.array(all_embeddings)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()  # Normalize

        weighted_embedding = np.average(all_embeddings, axis=0, weights=all_weights)
        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)

        # Create steering vectors using the vector extractor
        if self.vector_extractor is None:
            if self.llm is None:
                return None
            embedding_dim = len(weighted_embedding)
            hidden_dim = self.llm.hidden_size
            self.vector_extractor = SteeringVectorExtractor(
                source_dim=embedding_dim,
                target_dim=hidden_dim,
                projection_type="random",
                device=self.device,
            )

        dynamic_vectors = self.vector_extractor.create_multi_layer_vectors(
            embedding=weighted_embedding,
            n_layers=self.llm.num_layers,
        )

        return dynamic_vectors

    def apply_dynamic_steering(
        self,
        dynamic_vectors: Dict[int, torch.Tensor],
        blend_ratio: float = 0.5,
    ):
        """
        Apply dynamic steering vectors, optionally blending with global steering.

        Args:
            dynamic_vectors: Steering vectors computed from retrieved content
            blend_ratio: How much to blend dynamic vs global (0=all global, 1=all dynamic)
        """
        if dynamic_vectors is None or self.llm is None:
            return

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
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering vectors from Quran text.

        Args:
            chunk_by: How to chunk the Quran text
            cache_path: Path to cache/load embeddings
            use_cached: Whether to use cached embeddings if available

        Returns:
            Dictionary of steering vectors per layer
        """
        if self.embedder is None or self.llm is None:
            self.load_models()

        # Try to load cached embeddings
        if cache_path and use_cached:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print("Loading cached Quran embeddings...")
                self.quran_embeddings = self.embedder.load_cached_embeddings(cache_path)
            else:
                self.quran_embeddings = self.embedder.create_quran_embeddings(
                    file_path=self.quran_path,
                    chunk_by=chunk_by,
                    save_path=cache_path,
                )
        else:
            self.quran_embeddings = self.embedder.create_quran_embeddings(
                file_path=self.quran_path,
                chunk_by=chunk_by,
                save_path=cache_path,
            )

        # Get dimensions
        embedding_dim = self.quran_embeddings["embeddings"].shape[1]
        hidden_dim = self.llm.hidden_size
        num_layers = self.llm.num_layers

        print(f"Embedding dim: {embedding_dim}, LLM hidden dim: {hidden_dim}")

        # Create vector extractor
        self.vector_extractor = SteeringVectorExtractor(
            source_dim=embedding_dim,
            target_dim=hidden_dim,
            projection_type="random",  # Fast and effective
            device=self.device,
        )

        # Create steering vectors from mean Quran embedding
        mean_embedding = self.quran_embeddings["mean_embedding"]
        self.steering_vectors = self.vector_extractor.create_multi_layer_vectors(
            embedding=mean_embedding,
            n_layers=num_layers,
        )

        # Apply to LLM
        self._apply_steering()

        return self.steering_vectors

    def prepare_verse_steering(
        self,
        verse_indices: List[int],
        combine_method: Literal["mean", "max", "concat"] = "mean",
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering from specific verses.

        Args:
            verse_indices: Indices of verses to use
            combine_method: How to combine verse embeddings

        Returns:
            Steering vectors per layer
        """
        if self.quran_embeddings is None:
            raise ValueError("Run prepare_quran_steering first")

        embeddings = self.quran_embeddings["embeddings"]
        selected = embeddings[verse_indices]

        if combine_method == "mean":
            combined = np.mean(selected, axis=0)
        elif combine_method == "max":
            combined = np.max(selected, axis=0)
        else:
            combined = np.mean(selected, axis=0)  # Default to mean

        # Normalize
        combined = combined / np.linalg.norm(combined)

        # Create vectors
        self.steering_vectors = self.vector_extractor.create_multi_layer_vectors(
            embedding=combined,
            n_layers=self.llm.num_layers,
        )

        self._apply_steering()
        return self.steering_vectors

    def prepare_thematic_steering(
        self,
        theme_query: str,
        top_k: int = 10,
    ) -> Dict[int, torch.Tensor]:
        """
        Steer using verses most similar to a theme query.

        Args:
            theme_query: Text describing the theme (e.g., "mercy", "justice")
            top_k: Number of most similar verses to use

        Returns:
            Steering vectors per layer
        """
        if self.quran_embeddings is None or self.embedder is None:
            raise ValueError("Run prepare_quran_steering first")

        # Embed the query
        query_embedding = self.embedder.create_embeddings([theme_query])[0]

        # Find most similar verses
        embeddings = self.quran_embeddings["embeddings"]
        similarities = embeddings @ query_embedding

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        print(f"Top {top_k} verses for theme '{theme_query}':")
        for i, idx in enumerate(top_indices[:3]):  # Show top 3
            print(f"  {i+1}. {self.quran_embeddings['texts'][idx][:50]}...")

        return self.prepare_verse_steering(list(top_indices))

    def _apply_steering(self):
        """Apply current steering vectors to LLM."""
        if self.steering_vectors is None or self.llm is None:
            return

        # Determine which layers to steer
        target_layers = self.config.target_layers
        if target_layers is None:
            # Auto-select based on distribution
            num_layers = self.llm.num_layers
            if self.config.layer_distribution == "focused":
                # Focus on specific layer
                center = int(self.config.focus_layer * num_layers)
                target_layers = list(range(max(0, center - 2), min(num_layers, center + 3)))
            elif self.config.layer_distribution == "bell":
                # Use middle third of layers
                start = num_layers // 3
                end = 2 * num_layers // 3
                target_layers = list(range(start, end))
            else:
                # All layers
                target_layers = list(range(num_layers))

        # Scale vectors and apply
        for layer_idx in target_layers:
            if layer_idx not in self.steering_vectors:
                continue

            vector = self.steering_vectors[layer_idx]

            # Apply layer-specific scaling
            if self.config.layer_distribution == "bell":
                # Bell curve scaling
                center = self.llm.num_layers / 2
                scale = np.exp(-0.5 * ((layer_idx - center) / (self.llm.num_layers / 4)) ** 2)
            else:
                scale = 1.0

            scaled_vector = vector * scale * self.config.coefficient

            self.llm.register_steering_hook(
                layer_idx=layer_idx,
                steering_vector=scaled_vector,
                coefficient=1.0,  # Already scaled
                injection_mode=self.config.injection_mode,
            )

    def set_steering_strength(self, coefficient: float):
        """Adjust steering strength without recomputing vectors."""
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
        dynamic_blend_ratio: float = 0.5,
        **kwargs,
    ) -> str:
        """
        Generate text with Quran-influenced steering.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            mra_mode: Whether to use Multi-Resolution Analysis reasoning
            use_domain_bridges: Whether to generate domain bridges for better retrieval
            use_dynamic_steering: Whether to dynamically steer based on retrieved content
            dynamic_blend_ratio: Blend ratio between dynamic and global steering (0-1)
        """
        if self.llm is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        final_prompt = prompt

        if mra_mode:
            if self.knowledge_base is None:
                self.initialize_knowledge_base()

            # 1. Generate Domain Bridges
            bridge_queries = []
            if use_domain_bridges:
                bridge_queries = self.generate_domain_bridges(prompt, max_bridges=3)

            # 2. Retrieve Multi-Resolution Context (with bridges and embeddings)
            if bridge_queries:
                results = self.knowledge_base.query_with_bridges(
                    original_query=prompt,
                    bridge_queries=bridge_queries,
                    n_results=3,
                    include_embeddings=use_dynamic_steering
                )
                print(f"--- Domain Bridges Applied: {bridge_queries} ---")
            else:
                results = self.knowledge_base.query_multiresolution(
                    prompt,
                    n_results=3,
                    include_embeddings=use_dynamic_steering
                )

            # 3. Apply Dynamic Steering (steer towards retrieved content)
            if use_dynamic_steering:
                dynamic_vectors = self.compute_dynamic_steering(results)
                if dynamic_vectors:
                    self.apply_dynamic_steering(dynamic_vectors, blend_ratio=dynamic_blend_ratio)
                    print(f"--- Dynamic Steering Applied (blend={dynamic_blend_ratio}) ---")

            # 4. Construct MRA Prompt
            limit = 600  # Char limit per section to avoid context overflow

            verses_txt = "\n".join([f"- {r['content'][:limit]}" for r in results['verse']])
            passages_txt = "\n".join([f"- {r['content'][:limit]}" for r in results['passage']])
            surahs_txt = "\n".join([f"- {r['content'][:limit]}" for r in results['surah']])

            # Include domain bridges in the prompt for transparency
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
            print("--- MRA Context Injected ---")

        return self.llm.generate(
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def generate_unsteered(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """Generate text without steering."""
        with self.llm.steering_disabled():
            return self.llm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Compare steered vs unsteered outputs.

        Returns:
            Tuple of (steered_output, unsteered_output)
        """
        return self.llm.compare_outputs(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def batch_compare(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> List[Tuple[str, str]]:
        """Compare outputs for multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.compare(prompt, max_new_tokens=max_new_tokens))
        return results

    def analyze_effect(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict:
        """
        Analyze the steering effect across test prompts.

        Returns:
            Analysis dictionary with statistics
        """
        comparisons = self.batch_compare(test_prompts, max_new_tokens)

        analysis = {
            "prompts": test_prompts,
            "steered_outputs": [c[0] for c in comparisons],
            "unsteered_outputs": [c[1] for c in comparisons],
            "avg_length_steered": np.mean([len(c[0]) for c in comparisons]),
            "avg_length_unsteered": np.mean([len(c[1]) for c in comparisons]),
        }

        return analysis

    def save_steering_vectors(self, path: Union[str, Path]):
        """Save computed steering vectors."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        vectors_np = {
            str(k): v.cpu().numpy()
            for k, v in self.steering_vectors.items()
        }

        np.savez(
            path,
            **vectors_np,
            config_coefficient=self.config.coefficient,
            config_injection_mode=self.config.injection_mode,
        )

    def load_steering_vectors(self, path: Union[str, Path]):
        """Load pre-computed steering vectors."""
        data = np.load(path)

        self.steering_vectors = {}
        for key in data.files:
            if key.startswith("config_"):
                continue
            layer_idx = int(key)
            self.steering_vectors[layer_idx] = torch.tensor(
                data[key], device=self.device
            )

        if "config_coefficient" in data.files:
            self.config.coefficient = float(data["config_coefficient"])
        if "config_injection_mode" in data.files:
            self.config.injection_mode = str(data["config_injection_mode"])

        if self.llm is not None:
            self._apply_steering()


class ContrastiveQuranSteerer(QuranSteerer):
    """
    Steers LLMs using contrastive activation addition.

    Creates steering vectors by contrasting model activations on
    Quran-influenced vs baseline prompts.
    """

    def prepare_contrastive_steering(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        target_layer: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering using contrastive prompt pairs.

        Args:
            positive_prompts: Prompts representing desired behavior
            negative_prompts: Prompts representing baseline/contrast
            target_layer: Specific layer to target (None = middle layer)

        Returns:
            Steering vectors
        """
        if self.llm is None:
            self.load_models()

        if target_layer is None:
            target_layer = self.llm.num_layers // 2

        # Extract activations for both sets
        pos_activations = []
        neg_activations = []

        for prompt in positive_prompts:
            acts = self.llm.extract_layer_activations(prompt, layers=[target_layer])
            pos_activations.append(acts[target_layer])

        for prompt in negative_prompts:
            acts = self.llm.extract_layer_activations(prompt, layers=[target_layer])
            neg_activations.append(acts[target_layer])

        # Compute contrastive vector
        caa_extractor = ContrastiveSteeringExtractor(
            target_dim=self.llm.hidden_size,
            device=self.device,
        )

        steering_vector = caa_extractor.compute_from_activations(
            positive_activations=pos_activations,
            negative_activations=neg_activations,
            layer_idx=target_layer,
        )

        # Create vectors for nearby layers
        self.steering_vectors = {}
        for i in range(max(0, target_layer - 2), min(self.llm.num_layers, target_layer + 3)):
            scale = 1.0 - 0.3 * abs(i - target_layer)
            self.steering_vectors[i] = steering_vector * scale

        self._apply_steering()
        return self.steering_vectors
