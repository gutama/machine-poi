"""
LLM Wrapper for Activation Steering

Provides hooks into transformer layers to enable activation steering
during inference without modifying model weights.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Union, Tuple, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from contextlib import contextmanager


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


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a model architecture."""
    model_name_lower = model_name.lower()

    for key in MODEL_CONFIGS:
        if key in model_name_lower:
            return MODEL_CONFIGS[key]

    # Default to llama-like architecture
    return MODEL_CONFIGS["llama"]


class ActivationHook:
    """Hook to capture and optionally modify activations."""

    def __init__(
        self,
        layer_idx: int,
        steering_vector: Optional[torch.Tensor] = None,
        coefficient: float = 1.0,
        injection_mode: str = "add",  # "add", "replace", "blend"
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
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward hook that captures and modifies activations."""
        # Capture activation
        self.captured_activation = output.detach().clone()

        if not self.enabled or self.steering_vector is None:
            return output

        # Ensure steering vector is on same device and dtype
        steering = self.steering_vector.to(output.device, output.dtype)

        # Apply steering based on mode
        if self.injection_mode == "add":
            # Add steering vector to all token positions
            # output shape: [batch, seq_len, hidden_dim]
            modified = output + steering * self.coefficient
        elif self.injection_mode == "replace":
            # Replace activation with steering vector
            modified = steering.unsqueeze(0).unsqueeze(0).expand_as(output)
        elif self.injection_mode == "blend":
            # Blend original and steering
            alpha = self.coefficient
            modified = (1 - alpha) * output + alpha * steering.unsqueeze(0).unsqueeze(0).expand_as(output)
        else:
            modified = output

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

    def __init__(
        self,
        model_name: str = "qwen3-0.6b",
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

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_path}")

        # Prepare loading arguments
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
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

        print(f"Model loaded. Hidden size: {self.hidden_size}, Layers: {self.num_layers}")

    @property
    def hidden_size(self) -> int:
        """Get model hidden dimension."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return getattr(self.model.config, self.config["hidden_size_attr"])

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return getattr(self.model.config, self.config["num_layers_attr"])

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer."""
        layer_path = self.config["layer_name_pattern"].format(layer_idx=layer_idx)

        # Navigate to the module
        module = self.model
        for part in layer_path.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
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

        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

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
