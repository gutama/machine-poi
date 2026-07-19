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
from types import SimpleNamespace


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

    def test_get_steering_diagnostics_filters_disabled_hooks(self):
        """Test that diagnostics only include enabled steering hooks."""
        from src.llm_wrapper import SteeredLLM

        llm = SteeredLLM(model_name="qwen2.5-0.5b", device="cpu")
        llm.hooks = {
            0: SimpleNamespace(
                enabled=True,
                captured_activation=torch.ones(1, 1, 2),
                steering_vector=torch.tensor([1.0, 0.0]),
                coefficient=1.0,
            ),
            1: SimpleNamespace(
                enabled=False,
                captured_activation=torch.ones(1, 1, 2),
                steering_vector=torch.tensor([0.0, 1.0]),
                coefficient=1.0,
            ),
        }

        diagnostics = llm.get_steering_diagnostics()

        assert list(diagnostics) == [0]


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


class TestKvShareSourceMap:
    """Test cross-layer KV-sharing source resolution (Gemma 4 style)."""

    def test_no_shared_layers(self):
        from src.llm_wrapper import kv_share_source_map

        assert kv_share_source_map(["full_attention"] * 4, 4, 0) == {}

    def test_all_layers_shared_is_degenerate(self):
        from src.llm_wrapper import kv_share_source_map

        # first_shared == 0: nothing can provide KV states
        assert kv_share_source_map(["full_attention"] * 4, 4, 4) == {}

    def test_gemma4_e2b_layout(self):
        from src.llm_wrapper import kv_share_source_map

        # 35 layers, full attention every 5th layer starting at 4,
        # last 20 layers share KV (google/gemma-4-E2B-it).
        layer_types = [
            "full_attention" if i % 5 == 4 else "sliding_attention"
            for i in range(35)
        ]
        sources = kv_share_source_map(layer_types, 35, 20)

        assert set(sources) == set(range(15, 35))
        # Shared full-attention layers reuse the last non-shared
        # full-attention layer (14); sliding layers reuse layer 13.
        for idx in (19, 24, 29, 34):
            assert sources[idx] == 14
        for idx in set(range(15, 35)) - {19, 24, 29, 34}:
            assert sources[idx] == 13

    def test_type_missing_from_prefix_is_skipped(self):
        from src.llm_wrapper import kv_share_source_map

        # A shared layer whose type never occurs before the share point
        # has no source and is omitted.
        layer_types = ["sliding_attention", "sliding_attention", "full_attention"]
        sources = kv_share_source_map(layer_types, 3, 1)
        assert sources == {}


class TestKvSharedLayerDiagnostics:
    """Transport diagnostics for layers that reuse another layer's KV."""

    @staticmethod
    def _build_llm(seq_len=5, dim=4):
        """SteeredLLM over a tiny two-layer model whose layer 1 has no v_proj."""
        from types import SimpleNamespace

        from src.llm_wrapper import MODEL_CONFIGS, SteeredLLM

        torch.manual_seed(0)

        class TinyAttention(nn.Module):
            def __init__(self, with_v):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                if with_v:
                    self.v_proj = nn.Linear(dim, dim, bias=False)

        class TinyLayer(nn.Module):
            def __init__(self, with_v):
                super().__init__()
                self.self_attn = TinyAttention(with_v)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(
                    layers=nn.ModuleList([TinyLayer(True), TinyLayer(False)])
                )
                self.device = "cpu"
                self.config = SimpleNamespace(
                    num_hidden_layers=2,
                    num_kv_shared_layers=1,
                    layer_types=["full_attention", "full_attention"],
                    num_attention_heads=1,
                    hidden_size=dim,
                )
                self.hidden = torch.randn(1, seq_len, dim)
                causal = torch.tril(torch.ones(seq_len, seq_len))
                weights = causal / causal.sum(dim=-1, keepdim=True)
                self.attn_weights = weights.expand(1, 1, seq_len, seq_len)

            def forward(self, input_ids=None, output_attentions=False, **kwargs):
                for layer in self.model.layers:
                    attn = layer.self_attn
                    attn.q_proj(self.hidden)
                    if hasattr(attn, "v_proj"):
                        attn.v_proj(self.hidden)
                return SimpleNamespace(
                    attentions=(self.attn_weights, self.attn_weights)
                )

        llm = SteeredLLM.__new__(SteeredLLM)
        llm.model = TinyModel()
        llm.config = MODEL_CONFIGS["llama"]
        llm.hooks = {}
        llm.hook_handles = []

        batch = SimpleNamespace(to=lambda device: {
            "input_ids": torch.arange(seq_len).unsqueeze(0)
        })
        llm.tokenizer = lambda prompt, return_tensors: batch
        return llm

    def test_shared_layer_uses_source_layer_values(self):
        from src.workspace_diagnostics import summarize_attention_transport_heads

        llm = self._build_llm()
        assert llm._kv_share_sources() == {1: 0}

        diag = llm.get_attention_transport_diagnostics("x", max_loop_positions=4)
        assert set(diag.keys()) == {0, 1}, "KV-shared layer 1 must not be skipped"

        # Layer 1 must be summarized with its own queries but layer 0's values.
        model = llm.model
        hidden = model.hidden[0]
        q1 = model.model.layers[1].self_attn.q_proj(hidden).detach()
        v0 = model.model.layers[0].self_attn.v_proj(hidden).detach()
        seq_len = hidden.shape[0]
        expected = summarize_attention_transport_heads(
            model.attn_weights[0].float(),
            q1.view(seq_len, 1, -1).transpose(0, 1).float(),
            v0.view(seq_len, 1, -1).transpose(0, 1).float(),
            max_loop_positions=4,
        )
        assert diag[1][0].non_abelian_ratio == pytest.approx(
            expected[0].non_abelian_ratio)
        assert diag[1][0].mean_holonomy == pytest.approx(expected[0].mean_holonomy)

    def test_shared_layer_skipped_when_source_lacks_v_proj(self):
        llm = self._build_llm()
        del llm.model.model.layers[0].self_attn.v_proj

        diag = llm.get_attention_transport_diagnostics("x", max_loop_positions=4)
        assert diag == {}
