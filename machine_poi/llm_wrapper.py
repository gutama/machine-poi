"""
LLM Wrapper for Activation Steering

Provides hooks into transformer layers to enable activation steering
during inference without modifying model weights.
"""

import importlib.util
import logging
import math
import re
import threading
from functools import wraps
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from contextlib import contextmanager

from .config import LLM_MODELS
from .workspace_diagnostics import SteeringStats


# Setup module logger
logger = logging.getLogger("machine_poi.llm_wrapper")



# Every supported architecture exposes decoder layers at model.layers.N with
# hidden_size/num_hidden_layers on its (text) config. _get_layer_module also
# tries the paths multimodal wrappers such as Gemma 4 use.
DECODER_LAYOUT = {
    "layer_name_pattern": "model.layers.{layer_idx}",
    "hidden_size_attr": "hidden_size",
    "num_layers_attr": "num_hidden_layers",
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
    """Return the decoder layout; all supported architectures share it."""
    return dict(DECODER_LAYOUT)


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



def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def synchronized(method):
    """Serialize all inference and hook mutation on one model instance."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._steering_lock:
            return method(self, *args, **kwargs)
    return wrapped


class ActivationHook:
    """Hook that steers a layer's output and keeps running statistics.

    ``stats`` accumulates per-token steering statistics while the hook is
    enabled. ``capture=True`` additionally keeps a copy of the latest
    hidden states in ``captured_activation`` (off by default: it clones the
    full tensor on every forward pass).
    """

    def __init__(
        self,
        layer_idx: int,
        steering_vector: Optional[torch.Tensor] = None,
        coefficient: float = 1.0,
        injection_mode: str = "add",  # "add", "replace", "blend", "clamp"
        capture: bool = False,
        rotor_artifact=None,
        rotor_max_angle: float = 0.0,
    ):
        if injection_mode not in {"add", "blend", "replace", "clamp", "rotor"}:
            raise ValueError("Unknown injection mode")
        if not math.isfinite(coefficient):
            raise ValueError("Steering coefficient must be finite")
        if injection_mode == "blend" and not 0 <= coefficient <= 1:
            raise ValueError("Blend coefficient must be in [0, 1]")
        if steering_vector is not None:
            if steering_vector.ndim != 1 or not torch.isfinite(steering_vector).all():
                raise ValueError("Steering vector must be a finite 1D tensor")
        self.rotor_artifact = None
        self.rotor_max_angle = rotor_max_angle
        if injection_mode == "rotor":
            from .rotor import RotorArtifact, RotorStats
            if not isinstance(rotor_artifact, RotorArtifact):
                raise ValueError("Rotor hook requires a validated frozen artifact")
            if steering_vector is not None or coefficient != 1.0:
                raise ValueError("Rotor angles are separate from additive vectors/coefficients")
            if not math.isfinite(rotor_max_angle) or not 0 <= rotor_max_angle <= math.pi:
                raise ValueError("Invalid rotor angle")
            self.rotor_artifact = rotor_artifact.copy()
            self.rotor_stats = RotorStats()
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.injection_mode = injection_mode
        self.capture = capture
        self.captured_activation = None
        self.stats = SteeringStats()
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

        if self.capture:
            self.captured_activation = hidden_states.detach().clone()

        if self.enabled and self.rotor_artifact is not None:
            from .rotor import rotate_hidden
            modified, diagnostics = rotate_hidden(hidden_states, self.rotor_artifact, self.rotor_max_angle)
            self.rotor_stats.update(diagnostics)
            return (modified,) + rest if rest is not None else modified
        if not self.enabled or self.steering_vector is None:
            return output

        if hidden_states.shape[-1] != self.steering_vector.shape[0]:
            raise ValueError("Steering vector does not match hidden dimension")
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

        self.stats.update(hidden_states, self.steering_vector, self.coefficient,
                          self.injection_mode, modified=modified)
        # Return in same format as input
        if rest is not None:
            return (modified,) + rest
        return modified

    def set_steering_vector(self, vector: torch.Tensor, coefficient: float = 1.0):
        """Update the steering vector."""
        validated = ActivationHook(self.layer_idx, vector, coefficient, self.injection_mode)
        self.steering_vector = validated.steering_vector
        self.coefficient = validated.coefficient

    def disable(self):
        """Disable steering (passthrough)."""
        self.enabled = False

    def enable(self):
        """Enable steering."""
        self.enabled = True


class SteeredLLM:
    """
    Wraps a HuggingFace LLM to enable activation steering.

    Registered aliases, checkpoints and reasoning settings come from
    ``machine_poi.config.LLM_MODELS``; any other Hugging Face path also works
    if it uses a supported decoder layout.
    """

    SUPPORTED_MODELS = {alias: spec["hf_path"] for alias, spec in LLM_MODELS.items()}
    REASONING_CONFIGS = {
        alias: dict(spec["reasoning"])
        for alias, spec in LLM_MODELS.items()
        if "reasoning" in spec
    }

    def __init__(
        self,
        model_name: str = "deepseek-r1-1.5b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
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
        if trust_remote_code and not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
            raise ValueError("Remote code requires an explicitly pinned commit revision")
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self._steering_lock = threading.RLock()
        self._handles_by_layer = {}
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
        # Effective decoding settings of the latest generate call.
        self.last_generation_settings: Dict[str, Any] = {}
        self.hook_handles: List = []

    @synchronized
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_path}")

        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
            "dtype": self.torch_dtype,  # Was torch_dtype, deprecated
        }

        if self.load_in_8bit or self.load_in_4bit:
            # Passing load_in_*bit directly to from_pretrained is deprecated.
            if not bitsandbytes_available():
                raise ImportError(
                    "4-bit and 8-bit loading need bitsandbytes: "
                    "pip install 'machine-poi[quantization]'"
                )
            if self.load_in_8bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
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
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
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

    @synchronized
    def register_steering_hook(
        self,
        layer_idx: int,
        steering_vector: Optional[torch.Tensor] = None,
        coefficient: float = 1.0,
        injection_mode: str = "add",
        capture: bool = False,
        *,
        rotor_artifact=None,
        rotor_max_angle: float = 0.0,
        experimental_rotor: bool = False,
    ) -> ActivationHook:
        """
        Register a steering hook at a specific layer.

        Args:
            layer_idx: Layer index to hook
            steering_vector: Vector to inject
            coefficient: Scaling coefficient
            injection_mode: How to inject the vector
            capture: Keep the latest hidden states in captured_activation

        Returns:
            The registered hook
        """
        if injection_mode == "rotor" and not experimental_rotor:
            raise ValueError("Rotor registration requires experimental_rotor=True")
        if injection_mode != "rotor" and (rotor_artifact is not None or rotor_max_angle != 0 or experimental_rotor):
            raise ValueError("Rotor options require rotor injection mode")
        if injection_mode == "rotor":
            from .rotor import RotorArtifact
            if not isinstance(rotor_artifact, RotorArtifact):
                raise ValueError("Rotor registration requires a validated artifact")
        if self.model is None:
            self.load_model()
        if rotor_artifact is not None and rotor_artifact.basis.shape[0] != self.hidden_size:
            raise ValueError("Rotor basis does not match model hidden dimension")
        layer = self._get_layer_module(layer_idx)
        if steering_vector is not None and steering_vector.shape != (self.hidden_size,):
            raise ValueError("Steering vector does not match model hidden dimension")
        hook = ActivationHook(
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            coefficient=coefficient,
            injection_mode=injection_mode,
            capture=capture,
            rotor_artifact=rotor_artifact,
            rotor_max_angle=rotor_max_angle,
        )

        if layer_idx in self._handles_by_layer:
            self._handles_by_layer.pop(layer_idx).remove()
        handle = layer.register_forward_hook(hook)
        self.hooks[layer_idx] = hook
        self._handles_by_layer[layer_idx] = handle
        self.hook_handles = list(self._handles_by_layer.values())

        return hook

    @synchronized
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

    @synchronized
    def clear_steering(self):
        """Remove all steering hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()
        self._handles_by_layer.clear()

    @synchronized
    def disable_steering(self):
        """Temporarily disable all steering."""
        for hook in self.hooks.values():
            hook.disable()

    @synchronized
    def enable_steering(self):
        """Re-enable steering."""
        for hook in self.hooks.values():
            hook.enable()

    @contextmanager
    def steering_disabled(self):
        """Restore the exact prior enabled state, including nested use."""
        with self._steering_lock:
            previous = [(hook, hook.enabled) for hook in self.hooks.values()]
            self.disable_steering()
            try:
                yield
            finally:
                for hook, enabled in previous:
                    hook.enabled = enabled

    @contextmanager
    def steering_session(self):
        """Temporarily mutate steering, restore on failure, serialize callers.

        This synchronous scope must never span an await. Async callers retrieve
        data first, then run the complete inference section inside this scope.
        Direct model access bypasses this contract.
        """
        with self._steering_lock:
            previous = [
                (i, h.steering_vector.detach().clone() if h.steering_vector is not None else None,
                 h.coefficient, h.injection_mode, h.enabled, h.capture,
                 h.rotor_artifact.copy() if h.rotor_artifact is not None else None,
                 h.rotor_max_angle)
                for i, h in self.hooks.items()
            ]
            try:
                yield
            finally:
                self.clear_steering()
                for i, vector, coefficient, mode, enabled, capture, rotor, angle in previous:
                    hook = self.register_steering_hook(i, vector, coefficient, mode, capture,
                        rotor_artifact=rotor, rotor_max_angle=angle, experimental_rotor=rotor is not None)
                    hook.enabled = enabled

    @synchronized
    def reset_steering_stats(self) -> None:
        """Start new running statistics on every hook."""
        for hook in self.hooks.values():
            hook.stats.reset()
            if hook.rotor_artifact is not None:
                hook.rotor_stats.reset()

    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get captured activations from a layer (hooks registered with capture=True)."""
        if layer_idx in self.hooks:
            return self.hooks[layer_idx].captured_activation
        return None

    @synchronized
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
        return {**summarize_steering_hooks(enabled_hooks), **{
            layer: hook.rotor_stats.summary() for layer, hook in enabled_hooks.items()
            if getattr(hook, "rotor_artifact", None) is not None
        }}

    @synchronized
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
        try:
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

    def format_prompt(
        self,
        prompt: str,
        reasoning_mode: bool = False,
        chat_template: Optional[bool] = None,
    ) -> Tuple[str, bool]:
        """Format a prompt as a single user turn with the tokenizer's chat template.

        ``chat_template=None`` applies the template when the tokenizer has one
        (instruct checkpoints), ``True`` requires one and ``False`` leaves the
        prompt unchanged. Returns the text and whether a template was applied.
        """
        has_template = bool(getattr(self.tokenizer, "chat_template", None))
        if chat_template and not has_template:
            raise ValueError("The tokenizer has no chat template")
        reasoning = self.reasoning_config if reasoning_mode else None
        think_prefix = bool(reasoning and reasoning.get("force_think_prefix"))
        if not (has_template if chat_template is None else chat_template):
            if think_prefix and not prompt.lstrip().startswith("<think>"):
                prompt = "<think>\n" + prompt
            return prompt, False

        options = {}
        if (self.reasoning_config or {}).get("mode") == "qwen3":
            # Qwen3 templates think by default; follow the requested mode.
            options["enable_thinking"] = bool(reasoning)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **options
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if think_prefix and not text.rstrip().endswith("<think>"):
            # DeepSeek-R1 documentation: start the response with <think>.
            text += "<think>\n"
        return text, True

    @synchronized
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        reasoning_mode: bool = False,
        seed: Optional[int] = None,
        chat_template: Optional[bool] = None,
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
            seed: Seed torch's generators before decoding, for paired comparisons
            chat_template: See format_prompt; None applies the tokenizer's
                template when it has one

        Returns:
            Generated text
        """
        retrieval_options = sorted({"mra_mode", "use_domain_bridges"} & kwargs.keys())
        if retrieval_options:
            raise TypeError(
                f"{', '.join(retrieval_options)}: retrieval options belong to "
                "QuranSteerer.generate/compare; SteeredLLM.generate does not retrieve"
            )
        if self.model is None:
            self.load_model()
        # Diagnostics describe this call: prefill plus every decode step.
        self.reset_steering_stats()

        # Model-specific reasoning settings from the registry
        reasoning = self.reasoning_config if reasoning_mode else None
        if reasoning:
            temperature = reasoning.get("temperature", temperature)
            top_p = reasoning.get("top_p", top_p)
            do_sample = True  # Reasoning models need sampling
            if "top_k" in reasoning:
                kwargs["top_k"] = reasoning["top_k"]
        elif reasoning_mode:
            # Generic reasoning mode for models without native support
            temperature = min(temperature, 0.6)
            do_sample = True
            if "step by step" not in prompt.lower():
                prompt += "\nLet's think step by step:\n"

        prompt, templated = self.format_prompt(prompt, reasoning_mode, chat_template)
        # Templated text already contains BOS and other special tokens.
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=not templated)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Sampling parameters only apply when sampling; greedy decoding
        # ignores them and transformers warns if they are passed.
        sampling = {"temperature": temperature, "top_p": top_p} if do_sample else {}
        self.last_generation_settings = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "seed": seed,
            "reasoning_mode": reasoning_mode,
            "chat_template": templated,
            **sampling,
            **{key: kwargs[key] for key in ("top_k",) if key in kwargs},
        }
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
                **kwargs,
            )

        # Decode only new tokens. Qwen3 reasoning output keeps its
        # <think>...</think> block for the caller to parse.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @synchronized
    def compare_outputs(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Compare outputs with and without steering on the same prompt and seed.

        Returns:
            Tuple of (steered_output, unsteered_output)
        """
        # Generate with steering
        steered = self.generate(prompt, max_new_tokens=max_new_tokens, seed=seed, **kwargs)

        # Generate without steering
        with self.steering_disabled():
            unsteered = self.generate(
                prompt, max_new_tokens=max_new_tokens, seed=seed, **kwargs
            )

        return steered, unsteered

    @synchronized
    def continuation_logprob(
        self,
        context: str,
        continuation: str,
        add_special_tokens: bool = True,
    ) -> Tuple[float, int]:
        """
        Sum log-probability of ``continuation`` following ``context``.

        Runs one forward pass under whatever steering hooks are enabled; wrap
        the call in :meth:`steering_disabled` to score under the unsteered
        model. The continuation is tokenized on its own, without special
        tokens, and appended to the context tokens. Pass
        ``add_special_tokens=False`` for a context that is already
        chat-templated, as generation does.

        Returns:
            (sum of log-probabilities, number of continuation tokens)
        """
        if self.model is None:
            self.load_model()
        context_ids = self.tokenizer(context, add_special_tokens=add_special_tokens)["input_ids"]
        if not context_ids:
            raise ValueError("continuation_logprob needs a non-empty context")
        continuation_ids = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if not continuation_ids:
            return 0.0, 0
        ids = torch.tensor([context_ids + continuation_ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0]
        start = len(context_ids) - 1
        log_probs = torch.log_softmax(logits[start:-1].float(), dim=-1)
        targets = torch.tensor(continuation_ids, device=log_probs.device).unsqueeze(-1)
        return float(log_probs.gather(-1, targets).sum()), len(continuation_ids)

    @synchronized
    def pooled_layer_means(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 8,
        exclude_special: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Mean unsteered hidden state of each text at each layer, in batches.

        With ``exclude_special`` (the default), BOS, EOS and other special
        token positions are left out of the mean: they carry large generic
        activations (attention sinks) shared by every text. Set it to False
        to average every non-padding position, as before.

        Texts are right-padded so real tokens see the same positions and
        context as when run alone; padding is excluded from the mean. Batches
        group texts of similar token length to limit padding, and results
        come back in the input order. Hooks pool inside the forward pass
        instead of copying full hidden states.

        Returns:
            Dict mapping layer index to a float32 CPU tensor [len(texts), hidden]
        """
        def mean(hidden, mask):
            weights = mask.to(hidden.dtype).unsqueeze(-1)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

        pooled, order = self._reduce_layer_states(
            texts, layers, batch_size, exclude_special, mean
        )
        restore = torch.argsort(torch.tensor(order))
        return {layer: torch.cat(chunks)[restore] for layer, chunks in pooled.items()}

    @synchronized
    def layer_token_norms(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 8,
        exclude_special: bool = True,
    ) -> Dict[int, float]:
        """
        Median unsteered per-token hidden-state norm at each layer.

        The median over every content token of every text is the scale that
        dose ratios refer to. Unlike the mean, one attention-sink token with a
        massive norm barely moves it.
        """
        def norms(hidden, mask):
            return hidden.norm(dim=-1)[mask.bool()]

        collected, _ = self._reduce_layer_states(
            texts, layers, batch_size, exclude_special, norms
        )
        return {layer: float(torch.cat(chunks).median()) for layer, chunks in collected.items()}

    def _reduce_layer_states(self, texts, layers, batch_size, exclude_special, reduce):
        """Run texts through the unsteered model and reduce each layer's output.

        ``reduce(hidden, mask)`` receives one batch's layer output and its
        content-token mask (padding, and special tokens when excluded, are 0)
        and returns a float tensor. Returns per-layer lists of the reduced
        batches and the text order they ran in.
        """
        texts = list(texts)
        if not texts:
            raise ValueError("Layer statistics need at least one text")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.model is None:
            self.load_model()
        layers = list(range(self.num_layers)) if layers is None else list(layers)

        reduced: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
        current = {}
        lengths = [len(ids) for ids in self.tokenizer(texts)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda index: lengths[index])

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                mask = current["mask"].to(hidden.device)
                reduced[layer_idx].append(reduce(hidden, mask).float().cpu())
            return hook

        handles = []
        padding_side = getattr(self.tokenizer, "padding_side", None)
        try:
            for layer_idx in layers:
                layer = self._get_layer_module(layer_idx)
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
            if padding_side is not None:
                self.tokenizer.padding_side = "right"
            with self.steering_disabled(), torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = [texts[index] for index in order[start:start + batch_size]]
                    encoded = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        return_special_tokens_mask=exclude_special,
                    )
                    mask = encoded["attention_mask"]
                    if exclude_special:
                        mask = mask * (1 - encoded["special_tokens_mask"])
                    current["mask"] = mask
                    self.model(
                        input_ids=encoded["input_ids"].to(self.model.device),
                        attention_mask=encoded["attention_mask"].to(self.model.device),
                    )
        finally:
            for handle in handles:
                handle.remove()
            if padding_side is not None:
                self.tokenizer.padding_side = padding_side
        return reduced, order

    @synchronized
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
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach().clone()
            return hook

        try:
            for layer_idx in layers:
                layer = self._get_layer_module(layer_idx)
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self.steering_disabled(), torch.no_grad():
                self.model(**inputs)
            return captured
        finally:
            for handle in handles:
                handle.remove()
