"""
Configuration for Machine-POI LLM Steering

Contains model configurations, hyperparameters, and presets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


# Supported LLM models with their configurations
LLM_MODELS = {
    # Primary targets (small, efficient models)
    "deepseek-r1-1.5b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "hidden_size": 1536,
        "num_layers": 28,
        "recommended_layers": list(range(10, 20)),
        "recommended_coefficient": 0.4,
        # Native reasoning mode: uses <think>...</think> blocks
        "reasoning_mode": "deepseek",
        "reasoning_temperature": 0.6,
        "reasoning_top_p": 0.95,
    },
    "phi4-mini": {
        "hf_path": "microsoft/Phi-4-mini-reasoning",
        "hidden_size": 3072,
        "num_layers": 32,
        "recommended_layers": list(range(12, 24)),
        "recommended_coefficient": 0.3,
        # Built for math reasoning, no special tokens
        "reasoning_mode": "phi",
        "reasoning_temperature": 0.8,
        "reasoning_top_p": 0.95,
    },
    "qwen3-0.6b": {
        "hf_path": "Qwen/Qwen3-0.6B",
        "hidden_size": 1024,
        "num_layers": 28,
        "recommended_layers": list(range(10, 20)),
        "recommended_coefficient": 0.5,
        # Native thinking mode: uses enable_thinking=True in chat template
        "reasoning_mode": "qwen3",
        "reasoning_temperature": 0.6,
        "reasoning_top_p": 0.95,
        "reasoning_top_k": 20,
    },
    "smollm3": {
        "hf_path": "HuggingFaceTB/SmolLM3-3B",
        "hidden_size": 2560,
        "num_layers": 32,
        "recommended_layers": list(range(12, 24)),
        "recommended_coefficient": 0.35,
        "reasoning_mode": None,  # No native reasoning
    },
    "gemma-270m": {
        "hf_path": "google/gemma-3-270m-it",
        "hidden_size": 1024,
        "num_layers": 18,
        "recommended_layers": list(range(6, 14)),
        "recommended_coefficient": 0.5,
        "reasoning_mode": None,  # No native reasoning
    },
    # Fallback models (more widely available)
    "qwen2.5-0.5b": {
        "hf_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "hidden_size": 896,
        "num_layers": 24,
        "recommended_layers": list(range(8, 18)),
        "recommended_coefficient": 0.5,
        "reasoning_mode": None,  # Standard instruct model
    },
    "smollm2-135m": {
        "hf_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "hidden_size": 576,
        "num_layers": 30,
        "recommended_layers": list(range(10, 22)),
        "recommended_coefficient": 0.6,
    },
    "smollm2-360m": {
        "hf_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "hidden_size": 960,
        "num_layers": 32,
        "recommended_layers": list(range(12, 24)),
        "recommended_coefficient": 0.5,
    },
}

# Supported embedding models
EMBEDDING_MODELS = {
    "bge-m3": {
        "hf_path": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "max_length": 8192,
        "supports_arabic": True,
        "memory_gb": 2.5,
    },
    "qwen-embedding": {
        "hf_path": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "embedding_dim": 3584,
        "max_length": 32768,
        "supports_arabic": True,
        "memory_gb": 15.0,
    },
    "multilingual-e5": {
        "hf_path": "intfloat/multilingual-e5-large-instruct",
        "embedding_dim": 1024,
        "max_length": 512,
        "supports_arabic": True,
        "memory_gb": 2.0,
    },
    "bge-large-zh": {
        "hf_path": "BAAI/bge-large-zh-v1.5",
        "embedding_dim": 1024,
        "max_length": 512,
        "supports_arabic": False,  # Primarily Chinese
        "memory_gb": 1.5,
    },
}


@dataclass
class SteeringPreset:
    """Preset configurations for different steering behaviors."""

    name: str
    description: str
    coefficient: float
    target_layers: Optional[List[int]]
    injection_mode: str
    chunk_by: str
    layer_distribution: str


STEERING_PRESETS = {
    "gentle": SteeringPreset(
        name="gentle",
        description="Subtle influence, minimal disruption to base model",
        coefficient=0.2,
        target_layers=None,  # Auto-select
        injection_mode="add",
        chunk_by="verse",
        layer_distribution="bell",
    ),
    "moderate": SteeringPreset(
        name="moderate",
        description="Balanced influence, noticeable but not overwhelming",
        coefficient=0.5,
        target_layers=None,
        injection_mode="add",
        chunk_by="verse",
        layer_distribution="bell",
    ),
    "strong": SteeringPreset(
        name="strong",
        description="Strong influence, significant effect on outputs",
        coefficient=0.8,
        target_layers=None,
        injection_mode="add",
        chunk_by="paragraph",
        layer_distribution="uniform",
    ),
    "focused": SteeringPreset(
        name="focused",
        description="Concentrated effect on specific middle layers",
        coefficient=0.6,
        target_layers=None,
        injection_mode="add",
        chunk_by="verse",
        layer_distribution="focused",
    ),
}


@dataclass
class ExperimentConfig:
    """Configuration for a steering experiment."""

    # Model selection
    llm_model: str = "qwen3-0.6b"
    embedding_model: str = "bge-m3"

    # Paths
    quran_path: str = "al-quran.txt"
    cache_dir: str = "vectors"

    # Steering parameters
    preset: str = "moderate"
    custom_coefficient: Optional[float] = None
    custom_layers: Optional[List[int]] = None

    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Hardware
    device: Optional[str] = None  # Auto-detect
    quantization: Optional[str] = None  # "4bit", "8bit", or None

    def get_preset(self) -> SteeringPreset:
        """Get the steering preset configuration."""
        return STEERING_PRESETS[self.preset]

    def get_llm_config(self) -> Dict:
        """Get LLM model configuration."""
        return LLM_MODELS.get(self.llm_model, {})

    def get_embedding_config(self) -> Dict:
        """Get embedding model configuration."""
        return EMBEDDING_MODELS.get(self.embedding_model, {})


# Test prompts for evaluating steering effects
TEST_PROMPTS = [
    "What is the meaning of life?",
    "Tell me about justice and mercy.",
    "How should we treat others?",
    "What gives life purpose?",
    "Explain the concept of forgiveness.",
    "What is truth?",
    "How do we find peace?",
    "What is wisdom?",
    "Tell me about compassion.",
    "What is righteousness?",
]

# Prompts to test for specific Quranic themes
THEMATIC_TEST_PROMPTS = {
    "mercy": [
        "Describe what mercy means.",
        "How should we show mercy to others?",
        "Tell me about compassion and kindness.",
    ],
    "justice": [
        "What is true justice?",
        "How should justice be administered?",
        "Tell me about fairness and equality.",
    ],
    "patience": [
        "Why is patience important?",
        "How do we develop patience?",
        "Tell me about endurance through hardship.",
    ],
    "gratitude": [
        "What is the importance of gratitude?",
        "How should we express thankfulness?",
        "Tell me about appreciation and blessing.",
    ],
    "guidance": [
        "How do we find the right path?",
        "What does it mean to be guided?",
        "Tell me about moral direction.",
    ],
}


def get_recommended_config(
    llm_model: str,
    embedding_model: str = "bge-m3",
    intensity: str = "moderate",
) -> ExperimentConfig:
    """
    Get recommended configuration for a given model combination.

    Args:
        llm_model: LLM model name
        embedding_model: Embedding model name
        intensity: Steering intensity ("gentle", "moderate", "strong")

    Returns:
        Configured ExperimentConfig
    """
    config = ExperimentConfig(
        llm_model=llm_model,
        embedding_model=embedding_model,
        preset=intensity,
    )

    # Apply model-specific recommendations
    if llm_model in LLM_MODELS:
        model_config = LLM_MODELS[llm_model]
        config.custom_coefficient = model_config.get("recommended_coefficient")
        config.custom_layers = model_config.get("recommended_layers")

    return config
