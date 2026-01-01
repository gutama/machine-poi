# Machine-POI: LLM Steering with Quran Text Embeddings

Steer small language models using semantic embeddings derived from Quranic text, based on activation engineering techniques.

## Overview

This project implements **Activation Addition (ActAdd)** and **Contrastive Activation Addition (CAA)** to steer LLM outputs by injecting steering vectors into intermediate layer activations during inference.

### Key Features

- **Embedding-based Steering**: Uses text embeddings from Quran verses to create semantic steering vectors
- **Multiple Embedding Models**: Supports BGE-M3, Qwen3-Embedding, and other multilingual models
- **Small LLM Support**: Optimized for efficient models like Qwen3-0.6B, SmolLM, Gemma 270M
- **Thematic Steering**: Steer toward specific themes (mercy, justice, patience, etc.)
- **Comparison Mode**: Compare steered vs baseline outputs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src import QuranSteerer

# Initialize steerer
steerer = QuranSteerer(
    llm_model="qwen2.5-0.5b",      # Small, efficient LLM
    embedding_model="bge-m3",       # Multilingual embeddings
)

# Load models
steerer.load_models()

# Prepare steering from Quran text
steerer.prepare_quran_steering()

# Generate with steering
output = steerer.generate("What is the meaning of justice?")
print(output)

# Compare with baseline
steered, baseline = steerer.compare("Tell me about mercy and compassion")
print("Steered:", steered)
print("Baseline:", baseline)
```

## Command Line Usage

```bash
# Basic usage
python main.py

# Interactive mode
python main.py --interactive

# Compare on test prompts
python main.py --compare

# Custom settings
python main.py --llm qwen3-0.6b --preset strong --coefficient 0.7

# Thematic steering
python main.py --theme mercy --prompt "How should we treat others?"

# Use clamp injection mode (recommended for strong steering)
python main.py --injection-mode clamp --coefficient 0.8
```

## Supported Models

### LLMs (Target Models)

| Model | Size | HuggingFace Path |
|-------|------|------------------|
| DeepSeek-R1-Distill-Qwen | 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Phi-4-mini-reasoning | 3.8B | `microsoft/Phi-4-mini-reasoning` |
| Qwen3 | 0.6B | `Qwen/Qwen3-0.6B` |
| SmolLM3 | 3B | `HuggingFaceTB/SmolLM3-3B` |
| Gemma 3 | 270M | `google/gemma-3-270m-it` |
| Qwen2.5-Instruct | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| SmolLM2-Instruct | 135M-360M | `HuggingFaceTB/SmolLM2-*-Instruct` |

### Embedding Models

| Model | Dim | Best For |
|-------|-----|----------|
| BGE-M3 | 1024 | Multilingual, Arabic support |
| Qwen3-Embedding | 3584 | High quality, large context |
| Multilingual-E5 | 1024 | Balanced multilingual |

## How It Works

### 1. Text Embedding Extraction

The Quran text is chunked (by verse, paragraph, or surah) and embedded using a multilingual model:

```python
embedder = QuranEmbeddings(model_name="bge-m3")
embeddings = embedder.create_quran_embeddings(
    file_path="al-quran.txt",
    chunk_by="verse"
)
mean_embedding = embeddings["mean_embedding"]  # "Quran vector"
```

### 2. Steering Vector Projection

The embedding is projected to match the LLM's hidden dimension:

```python
extractor = SteeringVectorExtractor(
    source_dim=1024,   # Embedding dim
    target_dim=896,    # LLM hidden dim
    projection_type="random"
)
steering_vector = extractor.project_embedding(mean_embedding)
```

### 3. Activation Injection

During inference, the steering vector is injected into intermediate layer activations:

```python
llm.register_steering_hook(
    layer_idx=12,
    steering_vector=steering_vector,
    coefficient=0.5,
    injection_mode="clamp"  # or "add", "blend", "replace"
)
```

### Injection Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `add` | Add steering vector to activations | Gentle steering |
| `blend` | Interpolate between original and steering | Balanced control |
| `replace` | Replace activations entirely | Maximum effect |
| `clamp` | Remove existing projection, then add controlled amount | **Recommended for strong steering** |

> **Tip**: The `clamp` mode (inspired by [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama)) often produces more stable outputs than `add` when using higher coefficients, as it prevents over-biasing the model.

## Steering Presets

| Preset | Coefficient | Description |
|--------|-------------|-------------|
| `gentle` | 0.2 | Subtle influence |
| `moderate` | 0.5 | Balanced effect |
| `strong` | 0.8 | Strong influence |
| `focused` | 0.6 | Concentrated on middle layers |

## Advanced Usage

### Thematic Steering

Steer toward specific Quranic themes:

```python
# Find verses most similar to a theme
steerer.prepare_thematic_steering(
    theme_query="mercy and compassion",
    top_k=10
)
```

### Contrastive Steering

Use contrastive activation addition:

```python
from src import ContrastiveQuranSteerer

steerer = ContrastiveQuranSteerer(llm_model="qwen3-0.6b")
steerer.prepare_contrastive_steering(
    positive_prompts=["Speaking with wisdom and mercy..."],
    negative_prompts=["Speaking harshly..."]
)
```

### Custom Layer Selection

```python
steerer.config.target_layers = [10, 11, 12, 13, 14]  # Specific layers
steerer.config.coefficient = 0.6
steerer.config.injection_mode = "clamp"  # "add", "blend", "replace", "clamp"
```

## Project Structure

```
machine-poi/
├── al-quran.txt              # Quran text (Arabic)
├── main.py                   # CLI entry point
├── config.py                 # Model configs and presets
├── requirements.txt          # Dependencies
├── src/
│   ├── __init__.py
│   ├── quran_embeddings.py   # Embedding extraction
│   ├── steering_vectors.py   # Vector projection
│   ├── llm_wrapper.py        # LLM hooks and steering
│   └── steerer.py            # High-level interface
└── vectors/                  # Cached embeddings/vectors
```

## References

- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [Eiffel Tower LLaMA Demo](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama)
- [llm_steer Library](https://github.com/Mihaiii/llm_steer)
- [CAA Implementation](https://github.com/nrimsky/CAA)

## License

MIT License
