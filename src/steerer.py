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
        **kwargs,
    ) -> str:
        """
        Generate text with Quran-influenced steering.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.llm is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        return self.llm.generate(
            prompt=prompt,
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
