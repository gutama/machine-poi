"""
LLM Wrapper for Activation Steering

Provides hooks into transformer layers to enable activation steering
during inference without modifying model weights.
"""

import gc
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from contextlib import contextmanager


# Setup module logger
logger = logging.getLogger("machine_poi.llm_wrapper")



# Model configurations for supported architectures
MODEL_CONFIGS = {
    "deepseek": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",  # Where to inject
    },
    "qwen": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
    "phi": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
    "gemma": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
    "smollm": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
    "llama": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
    "mistral": {
        "layer_name_pattern": "model.layers.{layer_idx}",
        "hidden_size_attr": "hidden_size",
        "num_layers_attr": "num_hidden_layers",
        "residual_stream": "post_attention_layernorm",
    },
}



class LLMWrapperError(Exception):
    """Base exception for LLM wrapper errors."""
    pass


class LayerIndexError(LLMWrapperError):
    """Raised when an invalid layer index is specified."""
    pass


class ModelNotLoadedError(LLMWrapperError):
    """Raised when model is not loaded but required."""
    pass


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a model architecture."""
    model_name_lower = model_name.lower()

    for key in MODEL_CONFIGS:
        if key in model_name_lower:
            return MODEL_CONFIGS[key]

    # Default to llama-like architecture
    return MODEL_CONFIGS["llama"]


def kv_share_source_map(
    layer_types: List[str], num_layers: int, num_kv_shared_layers: int
) -> Dict[int, int]:
    """
    Map each KV-shared layer index to the layer whose key/value states it
    reuses.

    Architectures with cross-layer KV sharing (Gemma 4 / Gemma 3n) compute
    no k/v projections in their last num_kv_shared_layers layers; each such
    layer attends over the KV states produced by the LAST non-shared layer
    of the same attention type (mirrors store_full_length_kv in the
    transformers Gemma 4 implementation).

    Returns an empty dict when the model has no shared layers.
    """
    first_shared = num_layers - num_kv_shared_layers
    if num_kv_shared_layers <= 0 or first_shared <= 0:
        return {}
    # Configs are untrusted: bound the scan to the layer types actually
    # provided, and precompute each type's last non-shared occurrence.
    last_seen = {
        layer_type: idx
        for idx, layer_type in enumerate(layer_types[:first_shared])
    }
    return {
        layer_idx: last_seen[layer_types[layer_idx]]
        for layer_idx in range(first_shared, min(num_layers, len(layer_types)))
        if layer_types[layer_idx] in last_seen
    }



class ActivationHook:
    """Hook to capture and optionally modify activations."""

    def __init__(
        self,
        layer_idx: int,
        steering_vector: Optional[torch.Tensor] = None,
        coefficient: float = 1.0,
        injection_mode: str = "add",  # "add", "replace", "blend", "clamp"
    ):
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.injection_mode = injection_mode
        self.captured_activation = None
        self.enabled = True

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple],
    ) -> Union[torch.Tensor, Tuple]:
        """Forward hook that captures and modifies activations."""
        # Handle tuple outputs (common in HuggingFace models)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Capture activation
        self.captured_activation = hidden_states.detach().clone()

        if not self.enabled or self.steering_vector is None:
            return output

        # Ensure steering vector is on same device and dtype
        steering = self.steering_vector.to(hidden_states.device, hidden_states.dtype)

        # Apply steering based on mode
        if self.injection_mode == "add":
            # Add steering vector to all token positions
            # hidden_states shape: [batch, seq_len, hidden_dim]
            modified = hidden_states + steering * self.coefficient
        elif self.injection_mode == "replace":
            # Replace activation with steering vector
            modified = steering.unsqueeze(0).unsqueeze(0).expand_as(hidden_states)
        elif self.injection_mode == "blend":
            # Blend original and steering
            alpha = self.coefficient
            modified = (1 - alpha) * hidden_states + alpha * steering.unsqueeze(0).unsqueeze(0).expand_as(hidden_states)
        elif self.injection_mode == "clamp":
            # Clamp the activation along the steering direction.
            #
            # Intuition: remove the current projection on the direction, then add back a controlled amount.
            # This can be more stable than naive addition when steering is strong.
            v = steering
            v = v / (v.norm() + 1e-8)
            # projection of each token hidden state onto v: shape [batch, seq_len]
            proj_coeff = torch.einsum("bsh,h->bs", hidden_states, v)
            proj = proj_coeff.unsqueeze(-1) * v
            modified = hidden_states - proj + (self.coefficient * v)
        else:
            modified = hidden_states

        # Return in same format as input
        if rest is not None:
            return (modified,) + rest
        return modified

    def set_steering_vector(self, vector: torch.Tensor, coefficient: float = 1.0):
        """Update the steering vector."""
        self.steering_vector = vector
        self.coefficient = coefficient

    def disable(self):
        """Disable steering (passthrough)."""
        self.enabled = False

    def enable(self):
        """Enable steering."""
        self.enabled = True


class SteeredLLM:
    """
    Wraps a HuggingFace LLM to enable activation steering.

    Supports:
    - DeepSeek-R1-Distill-Qwen-1.5B
    - Microsoft Phi-4-mini-reasoning
    - Qwen3-0.6B
    - SmolLM3
    - Gemma 3 270M
    """

    SUPPORTED_MODELS = {
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "phi4-mini": "microsoft/Phi-4-mini-reasoning",
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "smollm3": "HuggingFaceTB/SmolLM3-3B",
        "gemma-270m": "google/gemma-3-270m-it",
        # Fallbacks/alternatives
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }

    # Model reasoning configurations (from official documentation)
    REASONING_CONFIGS = {
        "deepseek-r1-1.5b": {
            "mode": "deepseek",  # Uses <think>...</think> blocks
            "temperature": 0.6,
            "top_p": 0.95,
            "force_think_prefix": True,  # Enforce <think>\n at start
        },
        "phi4-mini": {
            "mode": "phi",  # Math-focused, no special tokens
            "temperature": 0.8,
            "top_p": 0.95,
            "force_think_prefix": False,
        },
        "qwen3-0.6b": {
            "mode": "qwen3",  # Native enable_thinking in chat template
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "force_think_prefix": False,  # Handled by tokenizer
        },
    }

    def __init__(
        self,
        model_name: str = "deepseek-r1-1.5b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the steered LLM.

        Args:
            model_name: Short name or HuggingFace model path
            device: Device to load model on
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            torch_dtype: Data type (default: auto)
        """
        self.model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.reasoning_config = self.REASONING_CONFIGS.get(model_name, None)

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.torch_dtype = torch_dtype or (
            torch.float16 if self.device != "cpu" else torch.float32
        )

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.config = None
        self.hooks: Dict[int, ActivationHook] = {}
        self.hook_handles: List = []

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_path}")

        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
            "dtype": self.torch_dtype,  # Was torch_dtype, deprecated
        }

        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model config
        self.config = get_model_config(self.model_path)

        logger.info(f"Model loaded. Hidden size: {self.hidden_size}, Layers: {self.num_layers}")

    def _text_config_attr(self, name: str):
        """
        Read a text-model attribute from the config, falling back to the
        nested text config for composite multimodal configs (e.g. Gemma 3/4
        *ForConditionalGeneration) where attributes like hidden_size live on
        config.text_config rather than the top level.
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        config = self.model.config
        if hasattr(config, name):
            return getattr(config, name)
        getter = getattr(config, "get_text_config", None)
        if callable(getter):
            text_config = getter()
            if hasattr(text_config, name):
                return getattr(text_config, name)
        raise AttributeError(
            f"Config {type(config).__name__} has no attribute {name!r} at the "
            "top level or on its text config"
        )

    def _kv_share_sources(self) -> Dict[int, int]:
        """
        Per-layer KV-sharing source map for the loaded model (empty for
        architectures without cross-layer KV sharing).
        """
        try:
            layer_types = self._text_config_attr("layer_types")
            num_shared = self._text_config_attr("num_kv_shared_layers")
        except AttributeError:
            return {}
        if not layer_types or not num_shared:
            return {}
        return kv_share_source_map(layer_types, self.num_layers, num_shared)

    @property
    def hidden_size(self) -> int:
        """Get model hidden dimension."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self._text_config_attr(self.config["hidden_size_attr"])

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self._text_config_attr(self.config["num_layers_attr"])

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer."""
        # Configured pattern first, then decoder-layer paths used by
        # multimodal wrappers (text tower nested under language_model).
        candidates = [
            self.config["layer_name_pattern"].format(layer_idx=layer_idx),
            f"model.language_model.layers.{layer_idx}",
            f"language_model.model.layers.{layer_idx}",
        ]
        for layer_path in candidates:
            module = self.model
            try:
                for part in layer_path.split("."):
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            return module
        raise LayerIndexError(
            f"Could not resolve layer {layer_idx}; tried paths: {candidates}"
        )

    def register_steering_hook(
        self,
        layer_idx: int,
        steering_vector: Optional[torch.Tensor] = None,
        coefficient: float = 1.0,
        injection_mode: str = "add",
    ) -> ActivationHook:
        """
        Register a steering hook at a specific layer.

        Args:
            layer_idx: Layer index to hook
            steering_vector: Vector to inject
            coefficient: Scaling coefficient
            injection_mode: How to inject the vector

        Returns:
            The registered hook
        """
        if self.model is None:
            self.load_model()

        layer = self._get_layer_module(layer_idx)
        hook = ActivationHook(
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            coefficient=coefficient,
            injection_mode=injection_mode,
        )

        handle = layer.register_forward_hook(hook)
        self.hooks[layer_idx] = hook
        self.hook_handles.append(handle)

        return hook

    def set_steering(
        self,
        steering_vectors: Dict[int, torch.Tensor],
        coefficient: float = 1.0,
    ):
        """
        Set steering vectors for multiple layers.

        Args:
            steering_vectors: Dict mapping layer indices to vectors
            coefficient: Global coefficient
        """
        for layer_idx, vector in steering_vectors.items():
            if layer_idx in self.hooks:
                self.hooks[layer_idx].set_steering_vector(vector, coefficient)
            else:
                self.register_steering_hook(layer_idx, vector, coefficient)

    def clear_steering(self):
        """Remove all steering hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()

    def disable_steering(self):
        """Temporarily disable all steering."""
        for hook in self.hooks.values():
            hook.disable()

    def enable_steering(self):
        """Re-enable steering."""
        for hook in self.hooks.values():
            hook.enable()

    @contextmanager
    def steering_disabled(self):
        """Context manager for temporarily disabling steering."""
        self.disable_steering()
        try:
            yield
        finally:
            self.enable_steering()

    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get captured activations from a layer."""
        if layer_idx in self.hooks:
            return self.hooks[layer_idx].captured_activation
        return None

    def get_steering_diagnostics(self) -> Dict[int, Any]:
        """
        Return workspace-inspired diagnostics for enabled steering hooks.

        Diagnostics are available after at least one forward pass has captured
        activations for registered hooks.
        """
        from .workspace_diagnostics import summarize_steering_hooks

        enabled_hooks = {
            layer_idx: hook
            for layer_idx, hook in self.hooks.items()
            if getattr(hook, "enabled", False)
        }
        return summarize_steering_hooks(enabled_hooks)

    def get_attention_transport_diagnostics(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
        eta: float = 1.0,
        max_loop_positions: int = 8,
    ) -> Dict[int, Dict[int, Any]]:
        """
        Discrete Cartan curvature diagnostics (non-abelian ratio ρ and
        holonomy) for attention heads on a single prompt.

        Runs one forward pass with attention outputs enabled, capturing
        per-head query/value projections, and summarizes each head's
        transport geometry via
        ``workspace_diagnostics.summarize_attention_transport_heads``.
        ρ ≈ 0 means the head's local transport generators nearly commute
        (weak path dependence); larger ρ and holonomy indicate
        order-sensitive context routing.

        Active steering hooks are left in place, so results reflect the
        current steering state; wrap the call in ``steering_disabled()``
        to measure the unsteered baseline.

        Args:
            prompt: Text to run the forward pass on.
            layers: Layer indices to analyze (default: all layers).
            eta: Transport step size η in T_t = exp(−η ω_t).
            max_loop_positions: Cap on positions used for holonomy loops.

        Returns:
            Dict mapping layer index → head index →
            AttentionTransportDiagnostics. KV-shared layers (cross-layer KV
            sharing, e.g. Gemma 4) are diagnosed using the value projections
            of the layer whose KV states they actually attend over; layers
            with no usable ``q_proj``/``v_proj`` at all are skipped.
        """
        from .workspace_diagnostics import summarize_attention_transport_heads

        if self.model is None:
            self.load_model()

        layer_indices = list(layers) if layers is not None else list(range(self.num_layers))

        captured_q: Dict[int, torch.Tensor] = {}
        captured_v: Dict[int, torch.Tensor] = {}
        handles = []

        def _capture(store: Dict[int, torch.Tensor], layer_idx: int):
            def hook(module: nn.Module, inputs: Tuple, output: torch.Tensor):
                store[layer_idx] = output.detach()
            return hook

        def _attn_module(layer_idx: int):
            layer = self._get_layer_module(layer_idx)
            return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)

        share_sources = self._kv_share_sources()
        hooked_layers = []
        v_source: Dict[int, int] = {}   # measured layer -> layer whose v_proj it uses
        v_hooked: Dict[int, bool] = {}  # v_proj hooks already registered, by source layer
        for layer_idx in layer_indices:
            attn = _attn_module(layer_idx)
            q_proj = getattr(attn, "q_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            source_idx = layer_idx
            if v_proj is None:
                # Cross-layer KV sharing: use the value projections of the
                # layer whose KV states this layer attends over.
                source_idx = share_sources.get(layer_idx)
                v_proj = getattr(_attn_module(source_idx), "v_proj", None) \
                    if source_idx is not None else None
            if q_proj is None or v_proj is None:
                logger.warning(
                    f"Layer {layer_idx}: could not resolve q_proj/v_proj, "
                    "either directly or through a KV-share source layer; "
                    "skipping transport diagnostics for this layer"
                )
                continue
            handles.append(q_proj.register_forward_hook(_capture(captured_q, layer_idx)))
            if source_idx not in v_hooked:
                handles.append(v_proj.register_forward_hook(_capture(captured_v, source_idx)))
                v_hooked[source_idx] = True
            v_source[layer_idx] = source_idx
            hooked_layers.append(layer_idx)

        if not hooked_layers:
            return {}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
        finally:
            for handle in handles:
                handle.remove()

        if outputs.attentions is None:
            logger.warning(
                "Model returned no attention weights (attention implementation "
                "may not support output_attentions); try loading with "
                "attn_implementation='eager'"
            )
            return {}

        num_heads = self._text_config_attr("num_attention_heads")
        diagnostics: Dict[int, Dict[int, Any]] = {}
        for layer_idx in hooked_layers:
            attn_weights = outputs.attentions[layer_idx][0].float().cpu()  # [heads, seq, seq]
            seq_len = attn_weights.shape[-1]
            q = captured_q[layer_idx][0].float().cpu()  # [seq, num_heads * head_dim]
            v = captured_v[v_source[layer_idx]][0].float().cpu()  # [seq, num_kv_heads * head_dim]

            if q.shape[-1] % num_heads != 0:
                logger.warning(
                    f"Layer {layer_idx}: q_proj dim {q.shape[-1]} not divisible by "
                    f"num_attention_heads {num_heads}; skipping transport diagnostics"
                )
                continue
            head_dim = q.shape[-1] // num_heads
            if v.shape[-1] % head_dim != 0:
                logger.warning(
                    f"Layer {layer_idx}: v_proj dim {v.shape[-1]} not divisible by "
                    f"head_dim {head_dim}; skipping transport diagnostics"
                )
                continue
            num_kv_heads = v.shape[-1] // head_dim
            q_heads = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
            v_heads = v.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
            if num_kv_heads != num_heads:
                if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
                    logger.warning(
                        f"Layer {layer_idx}: num_attention_heads {num_heads} not a "
                        f"multiple of num_key_value_heads {num_kv_heads}; skipping "
                        "transport diagnostics"
                    )
                    continue
                # Grouped-query attention: each KV head serves several query heads
                v_heads = v_heads.repeat_interleave(num_heads // num_kv_heads, dim=0)

            diagnostics[layer_idx] = summarize_attention_transport_heads(
                attn_weights,
                q_heads,
                v_heads,
                eta=eta,
                max_loop_positions=max_loop_positions,
            )
        return diagnostics

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        reasoning_mode: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text with steering applied.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample (vs greedy)
            reasoning_mode: Whether to enable native reasoning mode for supported models

        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        # Apply model-specific reasoning configuration
        if reasoning_mode and self.reasoning_config:
            config = self.reasoning_config
            mode = config.get("mode")
            
            # Use model-specific recommended parameters
            temperature = config.get("temperature", temperature)
            top_p = config.get("top_p", top_p)
            do_sample = True  # Reasoning models need sampling
            
            # Apply top_k if specified (Qwen3)
            if "top_k" in config:
                kwargs["top_k"] = config["top_k"]
            
            # Handle model-specific reasoning formats
            if mode == "deepseek":
                # DeepSeek-R1: Force thinking with <think> prefix
                # Per documentation: "enforce the model to initiate its response with <think>\n"
                if config.get("force_think_prefix"):
                    stripped = prompt.lstrip()
                    if not stripped.startswith("<think>"):
                        prompt = "<think>\n" + prompt
            
            elif mode == "qwen3":
                # Qwen3: Uses enable_thinking in chat template
                # Apply chat template with thinking enabled
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    # Fallback if enable_thinking not supported
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            
            elif mode == "phi":
                # Phi-4-mini-reasoning: Standard math reasoning
                # Uses chat format, add math prompt if relevant
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass  # Use raw prompt
            
            # Respect user-provided max_new_tokens; no forced bump
        
        elif reasoning_mode:
            # Generic reasoning mode for models without native support
            temperature = min(temperature, 0.6)
            do_sample = True
            if "step by step" not in prompt.lower():
                prompt += "\nLet's think step by step:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Filter out custom kwargs that shouldn't go to model.generate()
        custom_keys = ["mra_mode", "use_domain_bridges"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in custom_keys}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **filtered_kwargs,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # For Qwen3, optionally parse thinking content
        if reasoning_mode and self.reasoning_config and self.reasoning_config.get("mode") == "qwen3":
            # Output may contain <think>...</think> blocks
            # Return full output (including thinking) - user can parse if needed
            pass
        
        return output

    def compare_outputs(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Compare outputs with and without steering.

        Returns:
            Tuple of (steered_output, unsteered_output)
        """
        # Generate with steering
        steered = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

        # Generate without steering
        with self.steering_disabled():
            unsteered = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

        return steered, unsteered

    def extract_layer_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from specified layers for given text.

        Args:
            text: Input text
            layers: Layer indices to capture (default: all)

        Returns:
            Dict mapping layer indices to activation tensors
        """
        if self.model is None:
            self.load_model()

        if layers is None:
            layers = list(range(self.num_layers))

        # Register capture hooks
        captured = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                captured[layer_idx] = output.detach().clone()
            return hook

        for layer_idx in layers:
            layer = self._get_layer_module(layer_idx)
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        # Forward pass
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            self.model(**inputs)

        # Clean up
        for handle in handles:
            handle.remove()

        return captured
