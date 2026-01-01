# Machine-POI: LLM Steering with Quran Text Embeddings

Steer language model outputs using semantic embeddings derived from Quranic text. This project implements **activation engineering** techniques to inject Quran-derived "steering vectors" into LLM hidden states during inference.

## Overview

Machine-POI uses text embeddings from Quran verses to create semantic steering vectors that influence LLM behavior without fine-tuning. The steering is applied at inference time by modifying intermediate layer activations.

### Key Features

- **Quran Persona Mode**: Aggregate embeddings from all resolutions (verse, paragraph, surah) into a unified steering vector
- **Multi-Resolution Analysis (MRA)**: Dynamic context retrieval using ChromaDB knowledge base
- **Thematic Steering**: Steer toward specific themes (mercy, justice, patience, etc.)
- **Multiple Injection Modes**: `add`, `blend`, `replace`, and `clamp` (recommended for stability)
- **Comparison Mode**: Side-by-side comparison of steered vs baseline outputs
- **Model Comparison Tool**: Compare multiple models with the same prompt
- **Native Reasoning Modes**: DeepSeek-R1, Qwen3, Phi-4 reasoning support
- **8 Supported LLMs**: From 135M to 3.8B parameters

### Based On

- [Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248)
- [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681)
- [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama) — clamp injection mode inspired by this work

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- ~4GB RAM minimum (more for larger models)
- GPU recommended but not required

## Quick Start

### Python API

```python
from src import QuranSteerer

# Initialize
steerer = QuranSteerer(
    llm_model="qwen2.5-0.5b",
    embedding_model="bge-m3",
)

# Load models
steerer.load_models()

# Option 1: Standard Quran steering
steerer.prepare_quran_steering(chunk_by="verse")

# Option 2: Quran Persona (aggregates all resolutions - recommended)
steerer.prepare_quran_persona()

# Generate with steering
output = steerer.generate("What is the meaning of justice?")
print(output)

# Compare steered vs baseline
steered, baseline = steerer.compare("Tell me about mercy and compassion")
print("Steered:", steered)
print("Baseline:", baseline)
```

### Command Line

```bash
# Basic usage (uses qwen2.5-0.5b by default)
python main.py --interactive

# With Quran Persona mode (recommended)
python main.py --quran-persona --interactive

# Use clamp injection for more stable steering at higher coefficients
python main.py --quran-persona --injection-mode clamp --coefficient 0.8 --interactive

# Compare steered vs baseline on test prompts
python main.py --compare

# Thematic steering toward specific concepts
python main.py --theme mercy --prompt "How should we treat others?"

# Multi-Resolution Analysis mode
python main.py --mra --interactive

# Initialize ChromaDB knowledge base (required for MRA mode)
python main.py --init-db

# Enable native reasoning mode (uses model-specific config)
python main.py --llm deepseek-r1-1.5b --reasoning --prompt "What is wisdom?"

# Compare multiple models with the same prompt
./compare_models.py --models qwen3-0.6b deepseek-r1-1.5b --reasoning
./compare_models.py --list-models
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--llm MODEL` | LLM to steer (default: `qwen2.5-0.5b`) |
| `--embedding MODEL` | Embedding model for Quran text (default: `bge-m3`) |
| `--preset PRESET` | Steering preset: `gentle`, `moderate`, `strong`, `focused` |
| `--coefficient FLOAT` | Steering strength (0.0–1.0, default: 0.5) |
| `--injection-mode MODE` | How to inject: `add`, `blend`, `replace`, `clamp` |
| `--chunk-by TYPE` | Text chunking: `verse`, `paragraph`, `surah` |
| `--quran-persona` | Enable Quran Persona mode (aggregates all resolutions) |
| `--mra` | Enable Multi-Resolution Analysis with ChromaDB |
| `--theme THEME` | Steer toward a specific theme |
| `--interactive` | Interactive chat mode |
| `--compare` | Run comparison on test prompts |
| `--prompt TEXT` | Test a single prompt |
| `--init-db` | Initialize ChromaDB knowledge base |
| `--reasoning` | Enable native reasoning mode (model-specific) |
| `--device DEVICE` | Force device: `cuda`, `cpu`, `mps` |
| `--quantize MODE` | Quantization: `4bit`, `8bit` |

## Supported Models

### LLMs

| Model | Size | HuggingFace Path | Reasoning |
|-------|------|------------------|----------|
| `qwen2.5-0.5b` | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | — |
| `qwen3-0.6b` | 0.6B | `Qwen/Qwen3-0.6B` | ✅ `enable_thinking` |
| `smollm2-135m` | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | — |
| `smollm2-360m` | 360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | — |
| `smollm3` | 3B | `HuggingFaceTB/SmolLM3-3B` | — |
| `gemma-270m` | 270M | `google/gemma-3-270m-it` | — |
| `deepseek-r1-1.5b` | 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ✅ `<think>` blocks |
| `phi4-mini` | 3.8B | `microsoft/Phi-4-mini-reasoning` | ✅ Math reasoning |

### Embedding Models

| Model | Dim | Arabic Support | Memory |
|-------|-----|----------------|--------|
| `bge-m3` | 1024 | ✅ | ~2.5 GB |
| `multilingual-e5` | 1024 | ✅ | ~2.0 GB |
| `qwen-embedding` | 3584 | ✅ | ~15 GB |
| `bge-large-zh` | 1024 | ✅ | ~1.3 GB |

## How It Works

### 1. Text Embedding Extraction

The Quran text is chunked (by verse, paragraph, or surah) and embedded using a multilingual model:

```python
from src import QuranEmbeddings

embedder = QuranEmbeddings(model_name="bge-m3")
embedder.load_model()
data = embedder.create_quran_embeddings(
    file_path="al-quran.txt",
    chunk_by="verse"
)
mean_embedding = data["mean_embedding"]  # Shape: [1024]
```

### 2. Steering Vector Projection

The embedding is projected to match the LLM's hidden dimension:

```python
from src import SteeringVectorExtractor

extractor = SteeringVectorExtractor(
    source_dim=1024,   # Embedding model dim
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
    injection_mode="clamp"
)
```

### 4. Quran Persona Mode

Aggregates embeddings from all text resolutions into a unified steering vector:

```python
steerer.prepare_quran_persona()  # Loads verse + paragraph + surah embeddings
```

## Injection Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `add` | Add steering vector to activations | Gentle steering, lower coefficients |
| `blend` | Interpolate between original and steering | Balanced control |
| `replace` | Replace activations entirely | Maximum effect (use carefully) |
| `clamp` | Remove projection, add controlled amount | **Recommended for strong steering** |

The `clamp` mode is inspired by [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama). It first removes the existing projection of activations onto the steering direction, then adds a controlled amount. This prevents over-biasing and produces more fluent outputs at higher coefficients.

## Steering Presets

| Preset | Coefficient | Injection | Description |
|--------|-------------|-----------|-------------|
| `gentle` | 0.2 | add | Subtle influence |
| `moderate` | 0.5 | add | Balanced effect |
| `strong` | 0.8 | add | Strong influence |
| `focused` | 0.6 | add | Concentrated on middle layers |

## Advanced Usage

### Quran Persona Mode

The recommended way to use the steerer. Aggregates embeddings from verse, paragraph, and surah resolutions:

```bash
python main.py --quran-persona --injection-mode clamp --interactive
```

```python
steerer.prepare_quran_persona()
output = steerer.generate("What should guide our actions?")
```

### Multi-Resolution Analysis (MRA)

Uses ChromaDB for dynamic context retrieval. First initialize the database:

```bash
python main.py --init-db
python main.py --mra --interactive
```

### Thematic Steering

Steer toward specific Quranic themes by finding semantically similar verses:

```python
steerer.prepare_quran_steering(chunk_by="verse")
steerer.prepare_thematic_steering(
    theme_query="mercy and compassion",
    top_k=10  # Use top 10 most similar verses
)
```

### Contrastive Steering

Use contrastive activation addition with positive/negative examples:

```python
from src import ContrastiveQuranSteerer

steerer = ContrastiveQuranSteerer(llm_model="qwen3-0.6b")
steerer.prepare_contrastive_steering(
    positive_prompts=["Speaking with wisdom and mercy..."],
    negative_prompts=["Speaking harshly and without compassion..."]
)
```

### Custom Layer Selection

```python
steerer.config.target_layers = [10, 11, 12, 13, 14]
steerer.config.coefficient = 0.6
steerer.config.injection_mode = "clamp"
steerer.config.layer_distribution = "bell"  # or "uniform", "focused"
```

### Native Reasoning Mode

Models with native reasoning support use model-specific configurations:

| Model | Mode | Description |
|-------|------|-------------|
| `deepseek-r1-1.5b` | `<think>...</think>` | Forces think prefix, temp=0.6 |
| `qwen3-0.6b` | `enable_thinking` | Native chat template support, temp=0.6 |
| `phi4-mini` | Math reasoning | Optimized for mathematical reasoning, temp=0.8 |

```bash
# Enable reasoning with --reasoning flag
python main.py --llm deepseek-r1-1.5b --reasoning --prompt "What is justice?"

# Compare reasoning models
./compare_models.py --models deepseek-r1-1.5b qwen3-0.6b --reasoning
```

For models without native reasoning, a generic step-by-step prompting fallback is used.

## Project Structure

```
machine-poi/
├── main.py                   # CLI entry point
├── compare_models.py         # Model comparison tool
├── config.py                 # Model configs, presets, reasoning params
├── al-quran.txt              # Quran text (Arabic)
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py           # Package exports
│   ├── quran_embeddings.py   # Text embedding extraction
│   ├── steering_vectors.py   # Vector projection utilities
│   ├── llm_wrapper.py        # LLM hooks and steering injection
│   ├── steerer.py            # High-level QuranSteerer interface
│   └── knowledge_base.py     # ChromaDB for MRA mode
├── vectors/                  # Cached embeddings (auto-generated)
│   ├── quran_bge-m3_verse.npz
│   ├── quran_bge-m3_paragraph.npz
│   └── quran_bge-m3_surah.npz
└── quran_db/                 # ChromaDB storage (for MRA mode)
```

## Dependencies

- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.40.0` - HuggingFace model loading
- `sentence-transformers` - Embedding models
- `chromadb` - Vector database for MRA mode
- `bitsandbytes` - Quantization support (optional)
- `accelerate` - Model loading optimization

## References

- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama) — clamping approach
- [Neuronpedia](https://www.neuronpedia.org) — SAE exploration

## License

MIT License
