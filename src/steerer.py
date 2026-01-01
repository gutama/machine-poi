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
import textwrap

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

    # Injection mode: "add", "blend", "replace", "clamp"
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

    def compute_dynamic_steering(
        self,
        retrieved_results: Dict[str, List[Dict]],
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
        if resolution_weights is None:
            resolution_weights = {"verse": 0.5, "passage": 0.35, "surah": 0.15}

        # Collect text content to process
        texts_to_process = []
        
        for res_name, items in retrieved_results.items():
            res_weight = resolution_weights.get(res_name, 0.33)
            for item in items:
                text = item["content"]
                score = item.get("score", 0.5)
                # We'll pass the weight along
                texts_to_process.append({
                    "text": text,
                    "weight": res_weight * score
                })
        
        if not texts_to_process:
            return None

        # Compute activations
        layer_activations = {}
        processed_weights = []

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
        
        dynamic_vectors = {}
        for layer_idx, act_list in layer_activations.items():
            # Stack: [num_items, hidden_dim]
            stacked = torch.stack(act_list)
            
            # Weighted average
            weighted_mean = torch.sum(stacked * weights.unsqueeze(-1), dim=0)
            
            # Normalize
            weighted_mean = torch.nn.functional.normalize(weighted_mean, dim=-1)
            dynamic_vectors[layer_idx] = weighted_mean

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
        sample_size: int = 50,  # Number of verses to sample
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering vectors from Quran text using Mean Activation steering.
        """
        if self.embedder is None or self.llm is None:
            self.load_models()

        # Try to load cached steering vectors directly
        if cache_path and use_cached:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print("Loading cached steering vectors...")
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    # Handle both npz formats
                    if 'steering_vectors' in data:
                        # If saved as a single object
                        loaded_vecs = data['steering_vectors'].item()
                        self.steering_vectors = {
                            int(k): torch.tensor(v, device=self.device) 
                            for k, v in loaded_vecs.items()
                        }
                    else:
                        # If saved as kwargs
                        self.steering_vectors = {
                            int(k): torch.tensor(v, device=self.device) 
                            for k, v in data.items()
                        }
                    self._apply_steering()
                    return self.steering_vectors
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")

        # Load text
        texts = self.embedder.load_quran_text(self.quran_path, chunk_by=chunk_by)
        
        # Sample texts if too many
        if len(texts) > sample_size:
            print(f"Sampling {sample_size} verses/chunks from {len(texts)} total...")
            rng = np.random.RandomState(42)
            selected_texts = rng.choice(texts, size=sample_size, replace=False)
        else:
            selected_texts = texts

        print("Computing mean activations from Quran text...")
        
        layer_activations = {}
        
        for i, text in enumerate(selected_texts):
            if i % 10 == 0:
                print(f"Processing {i}/{len(selected_texts)}...")
                
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
            save_dict = {str(k): v.cpu().numpy() for k, v in self.steering_vectors.items()}
            np.savez(cache_path, **save_dict)
            print(f"Saved steering vectors to {cache_path}")

        # Apply to LLM
        self._apply_steering()

        return self.steering_vectors

    def prepare_verse_steering(
        self,
        verse_indices: List[int],
        combine_method: Literal["mean", "max", "concat"] = "mean",
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering from specific verses (using activations on demand).
        """
        if self.embedder is None:
            self.load_models(load_llm=False, load_embedder=True)

        texts = self.embedder.load_quran_text(self.quran_path, chunk_by="verse")
        selected_texts = [texts[i] for i in verse_indices]
        
        # Borrow dynamic steering logic to compute vectors
        # Dummy retrieval structure
        fake_retrieval = {"verse": [{"content": t, "score": 1.0} for t in selected_texts]}
        
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
        """
        if self.embedder is None:
            self.load_models(load_llm=False)

        # Embed the query
        query_embedding = self.embedder.create_embeddings([theme_query])[0]
        
        # Note: We need Quran embeddings for SIMILARITY SEARCH even if we use activations for STEERING
        # So we do need embeddings
        if self.quran_embeddings is None:
             self.quran_embeddings = self.embedder.create_quran_embeddings(
                file_path=self.quran_path,
                chunk_by="verse"
            )

        embeddings = self.quran_embeddings["embeddings"]
        similarities = embeddings @ query_embedding

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        print(f"Top {top_k} verses for theme '{theme_query}':")
        for i, idx in enumerate(top_indices[:3]):
            print(f"  {i+1}. {self.quran_embeddings['texts'][idx][:50]}...")

        return self.prepare_verse_steering(list(top_indices))

    def prepare_quran_persona(
        self,
        cache_dir: str = "vectors",
    ) -> Dict[int, torch.Tensor]:
        """
        Create a "Quran Persona" by aggregating all levels.
        """
        # Simply call prepared_quran_steering with a larger sample size and mixing resolutions
        # For simplicity, we'll just use the verse-level aggregation for now, as it's the base unit
        return self.prepare_quran_steering(
            chunk_by="verse",
            cache_path=Path(cache_dir) / f"quran_persona_activations.npz",
            sample_size=100
        )

    def _apply_steering(self):
        """Apply current steering vectors to LLM."""
        if self.steering_vectors is None or self.llm is None:
            return

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
        reasoning_mode: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text with Quran-influenced steering.
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

            # 2. Retrieve Multi-Resolution Context 
            if bridge_queries:
                results = self.knowledge_base.query_with_bridges(
                    original_query=prompt,
                    bridge_queries=bridge_queries,
                    n_results=3,
                    include_embeddings=False # We fetch content only, then compute act
                )
                print(f"--- Domain Bridges Applied: {bridge_queries} ---")
            else:
                results = self.knowledge_base.query_multiresolution(
                    prompt,
                    n_results=3,
                    include_embeddings=False
                )

            # 3. Apply Dynamic Steering (steer towards retrieved content)
            if use_dynamic_steering:
                dynamic_vectors = self.compute_dynamic_steering(results)
                if dynamic_vectors:
                    self.apply_dynamic_steering(dynamic_vectors, blend_ratio=dynamic_blend_ratio)
                    print(f"--- Dynamic Steering Applied (blend={dynamic_blend_ratio}) ---")
                    dynamic_applied = True

            # 4. Construct MRA Prompt - No Truncation
            # We trust the LLM's context window (usually sufficient for a few verses/passages)
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
            print("--- MRA Context Injected ---")

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
        with self.llm.steering_disabled():
            return self.llm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def compare(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[str, str]:
        """Compare steered vs unsteered outputs."""
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
        """Analyze the steering effect across test prompts."""
        comparisons = self.batch_compare(test_prompts, max_new_tokens)

        analysis = {
            "prompts": test_prompts,
            "steered_outputs": [c[0] for c in comparisons],
            "unsteered_outputs": [c[1] for c in comparisons],
            "avg_length_steered": np.mean([len(c[0]) for c in comparisons]),
            "avg_length_unsteered": np.mean([len(c[1]) for c in comparisons]),
        }
        return analysis


class ContrastiveQuranSteerer(QuranSteerer):
    """
    Steerer using Contrastive Activation Addition (CAA).
    
    Uses paired positive/negative examples to determine steering direction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_extractor = None

    def prepare_contrastive_steering(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
    ) -> Dict[int, torch.Tensor]:
        """
        Prepare steering from contrastive pairs.
        """
        if self.llm is None:
            self.load_models(load_embedder=False)
            
        print(f"Computing contrastive vectors from {len(positive_texts)} pairs...")
        
        # We need a new extractor for this
        if self.contrastive_extractor is None:
             self.contrastive_extractor = ContrastiveSteeringExtractor(
                hidden_dim=self.llm.hidden_size,
                device=self.device
            )
        
        # This is a placeholder as per user request to enable imports
        return {}
