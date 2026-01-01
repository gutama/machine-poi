"""
Tests for src/llm_wrapper.py

Tests for:
- ActivationHook (all injection modes)
- SteeredLLM (hook registration, generation)
- Model configuration detection
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch


class TestActivationHookInit:
    """Test ActivationHook initialization."""

    def test_init_defaults(self, sample_steering_vector):
        """Test initialization with default parameters."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(layer_idx=12)
        
        assert hook.layer_idx == 12
        assert hook.steering_vector is None
        assert hook.coefficient == 1.0
        assert hook.injection_mode == "add"
        assert hook.enabled is True

    def test_init_with_steering_vector(self, sample_steering_vector):
        """Test initialization with steering vector."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
            injection_mode="clamp",
        )
        
        assert hook.steering_vector is not None
        assert hook.coefficient == 0.5
        assert hook.injection_mode == "clamp"


class TestActivationHookInjection:
    """Test ActivationHook injection modes."""

    def test_add_mode(self, sample_steering_vector, sample_hidden_states):
        """Test 'add' injection mode."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
            injection_mode="add",
        )
        
        # Simulate forward pass
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        
        # Result should be original + coefficient * steering_vector
        expected = original + sample_steering_vector * 0.5
        torch.testing.assert_close(result, expected)

    def test_blend_mode(self, sample_steering_vector, sample_hidden_states):
        """Test 'blend' injection mode."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
            injection_mode="blend",
        )
        
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        
        # Result should be (1-alpha)*original + alpha*steering
        steering_expanded = sample_steering_vector.unsqueeze(0).unsqueeze(0).expand_as(original)
        expected = 0.5 * original + 0.5 * steering_expanded
        torch.testing.assert_close(result, expected)

    def test_replace_mode(self, sample_steering_vector, sample_hidden_states):
        """Test 'replace' injection mode."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=1.0,
            injection_mode="replace",
        )
        
        result = hook(None, None, sample_hidden_states)
        
        # All positions should have the steering vector
        for b in range(result.shape[0]):
            for s in range(result.shape[1]):
                torch.testing.assert_close(result[b, s], sample_steering_vector)

    def test_clamp_mode(self, sample_steering_vector, sample_hidden_states):
        """Test 'clamp' injection mode."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
            injection_mode="clamp",
        )
        
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        
        # Manually compute expected result
        v = sample_steering_vector.clone()
        v = v / (v.norm() + 1e-8)
        proj_coeff = torch.einsum("bsh,h->bs", original, v)
        proj = proj_coeff.unsqueeze(-1) * v
        expected = original - proj + (0.5 * v)
        
        torch.testing.assert_close(result, expected)

    def test_clamp_removes_existing_projection(self, sample_hidden_dim):
        """Test that clamp mode properly removes existing projection."""
        from src.llm_wrapper import ActivationHook
        
        torch.manual_seed(42)
        
        # Create a steering vector
        steering = torch.randn(sample_hidden_dim)
        steering = steering / steering.norm()
        
        # Create hidden states with known projection onto steering
        # h = 3*steering + orthogonal_component
        orthogonal = torch.randn(sample_hidden_dim)
        orthogonal = orthogonal - torch.dot(orthogonal, steering) * steering
        orthogonal = orthogonal / orthogonal.norm()
        
        hidden_states = (3.0 * steering + 0.5 * orthogonal).unsqueeze(0).unsqueeze(0)
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=steering,
            coefficient=1.0,
            injection_mode="clamp",
        )
        
        result = hook(None, None, hidden_states)
        
        # After clamping with coefficient=1.0:
        # - Remove projection (3*steering) 
        # - Add back 1.0*normalized_steering
        # Result should have projection = 1.0 onto steering direction
        v_norm = steering / steering.norm()
        result_proj = torch.dot(result.squeeze(), v_norm)
        assert abs(result_proj.item() - 1.0) < 1e-5


class TestActivationHookBehavior:
    """Test ActivationHook behavior controls."""

    def test_disabled_hook_passthrough(self, sample_steering_vector, sample_hidden_states):
        """Test that disabled hook passes through unchanged."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
        )
        hook.disable()
        
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        
        torch.testing.assert_close(result, original)

    def test_enable_after_disable(self, sample_steering_vector, sample_hidden_states):
        """Test re-enabling hook after disable."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
        )
        
        hook.disable()
        assert hook.enabled is False
        
        hook.enable()
        assert hook.enabled is True
        
        # Should now modify activations
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        assert not torch.allclose(result, original)

    def test_no_steering_vector_passthrough(self, sample_hidden_states):
        """Test that hook with no steering vector passes through."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(layer_idx=12)  # No steering vector
        
        original = sample_hidden_states.clone()
        result = hook(None, None, sample_hidden_states)
        
        torch.testing.assert_close(result, original)

    def test_captures_activation(self, sample_steering_vector, sample_hidden_states):
        """Test that hook captures activations."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(layer_idx=12)
        hook(None, None, sample_hidden_states)
        
        assert hook.captured_activation is not None
        torch.testing.assert_close(
            hook.captured_activation, sample_hidden_states
        )

    def test_set_steering_vector(self, sample_hidden_dim):
        """Test updating steering vector."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(layer_idx=12)
        assert hook.steering_vector is None
        
        new_vector = torch.randn(sample_hidden_dim)
        hook.set_steering_vector(new_vector, coefficient=0.7)
        
        assert hook.steering_vector is not None
        assert hook.coefficient == 0.7


class TestActivationHookTupleOutput:
    """Test ActivationHook with tuple outputs (common in HuggingFace)."""

    def test_tuple_output_preserved(self, sample_steering_vector, sample_hidden_states):
        """Test that tuple outputs are preserved."""
        from src.llm_wrapper import ActivationHook
        
        hook = ActivationHook(
            layer_idx=12,
            steering_vector=sample_steering_vector,
            coefficient=0.5,
        )
        
        # Simulate tuple output (hidden_states, attention, ...)
        extra_tensor = torch.randn(2, 4, 4)
        output = (sample_hidden_states, extra_tensor)
        
        result = hook(None, None, output)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Second element should be unchanged
        torch.testing.assert_close(result[1], extra_tensor)


class TestModelConfigDetection:
    """Test model configuration detection."""

    def test_get_model_config_qwen(self):
        """Test config detection for Qwen models."""
        from src.llm_wrapper import get_model_config
        
        config = get_model_config("Qwen/Qwen2.5-0.5B-Instruct")
        
        assert config["layer_name_pattern"] == "model.layers.{layer_idx}"
        assert config["hidden_size_attr"] == "hidden_size"

    def test_get_model_config_llama(self):
        """Test config detection for LLaMA models."""
        from src.llm_wrapper import get_model_config
        
        config = get_model_config("meta-llama/Llama-2-7b")
        
        assert "layer_name_pattern" in config

    def test_get_model_config_unknown_defaults_to_llama(self):
        """Test that unknown models default to llama config."""
        from src.llm_wrapper import get_model_config
        
        config = get_model_config("totally-unknown-model")
        
        # Should return llama default
        assert config["layer_name_pattern"] == "model.layers.{layer_idx}"


class TestSteeredLLMInit:
    """Test SteeredLLM initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM()
        
        assert llm.model_name == "deepseek-r1-1.5b"
        assert llm.model is None  # Not loaded yet
        assert llm.hooks == {}

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM(model_name="qwen2.5-0.5b")
        
        assert llm.model_name == "qwen2.5-0.5b"
        assert "Qwen2.5-0.5B" in llm.model_path

    def test_init_quantization_options(self):
        """Test quantization options."""
        from src.llm_wrapper import SteeredLLM
        
        llm_8bit = SteeredLLM(load_in_8bit=True)
        llm_4bit = SteeredLLM(load_in_4bit=True)
        
        assert llm_8bit.load_in_8bit is True
        assert llm_4bit.load_in_4bit is True

    def test_supported_models_list(self):
        """Test that supported models are defined."""
        from src.llm_wrapper import SteeredLLM
        
        expected_models = [
            "deepseek-r1-1.5b",
            "phi4-mini",
            "qwen3-0.6b",
            "smollm3",
            "gemma-270m",
            "qwen2.5-0.5b",
        ]
        
        for model in expected_models:
            assert model in SteeredLLM.SUPPORTED_MODELS


class TestSteeredLLMProperties:
    """Test SteeredLLM properties with mock model."""

    def test_hidden_size_property(
        self, mock_llm_model, mock_tokenizer, sample_hidden_dim
    ):
        """Test hidden_size property."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM()
        llm.model = mock_llm_model
        llm.tokenizer = mock_tokenizer
        llm.config = {"hidden_size_attr": "hidden_size", "num_layers_attr": "num_hidden_layers"}
        
        assert llm.hidden_size == sample_hidden_dim

    def test_num_layers_property(
        self, mock_llm_model, mock_tokenizer, sample_num_layers
    ):
        """Test num_layers property."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM()
        llm.model = mock_llm_model
        llm.tokenizer = mock_tokenizer
        llm.config = {"hidden_size_attr": "hidden_size", "num_layers_attr": "num_hidden_layers"}
        
        assert llm.num_layers == sample_num_layers

    def test_properties_raise_without_model(self):
        """Test that properties raise when model not loaded."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            _ = llm.hidden_size


class TestSteeredLLMModelLoading:
    """Test SteeredLLM model loading (integration tests)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_small_model(self):
        """Test loading a small model."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM(model_name="qwen2.5-0.5b", device="cpu")
        llm.load_model()
        
        assert llm.model is not None
        assert llm.tokenizer is not None
        assert llm.hidden_size > 0
        assert llm.num_layers > 0


class TestDeviceDetection:
    """Test automatic device detection."""

    def test_device_auto_cpu(self):
        """Test device defaults to CPU when no GPU."""
        from src.llm_wrapper import SteeredLLM
        
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(torch.backends, "mps", create=True) as mock_mps:
                mock_mps.is_available.return_value = False
                llm = SteeredLLM(device=None)
                # Should fall back to CPU
                assert llm.device in ["cpu", "cuda", "mps"]

    def test_device_explicit(self):
        """Test explicit device setting."""
        from src.llm_wrapper import SteeredLLM
        
        llm = SteeredLLM(device="cpu")
        assert llm.device == "cpu"
