# Machine-POI: LLM Steering with Quranic Semantic Embeddings

Steer language model outputs using semantic embeddings derived from Quranic text. This project implements **activation engineering** techniques to inject Quran-derived "steering vectors" into LLM hidden states during inference.

```
╔═══════════════════════════════════════════════════════════════════════╗
║                           Machine-POI                                  ║
║          LLM Steering with Quranic Semantic Embeddings                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Features:                                                             ║
║    • Multi-Resolution Analysis (Verse/Passage/Surah)                   ║
║    • Domain Bridging for Cross-Domain Analogies                        ║
║    • Quran Persona Steering                                            ║
║    • Contrastive Activation Addition (CAA)                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Based on: https://arxiv.org/abs/2308.10248 (ActAdd)                   ║
║            https://arxiv.org/abs/2312.06681 (CAA)                      ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## Overview

Machine-POI uses text embeddings from Quran verses to create semantic steering vectors that influence LLM behavior without fine-tuning. The steering is applied at inference time by modifying intermediate layer activations.

### Key Logic: Activation Steering

Unlike simple embedding projection (which can be random), this project uses **Mean Activation Steering**. We run representative samples of Quranic text through the LLM itself to capture the "Quranic Mindset" as a set of activation vectors. This ensures the steering is mathematically consistent with the model's internal representation.

### Key Features

- **Quran Persona Mode**: Aggregate activations from all resolutions (verse, paragraph, surah) into a unified steering vector
- **Multi-Resolution Analysis (MRA)**: Dynamic context retrieval using ChromaDB knowledge base
  - **Verse (Micro)**: Individual ayat (~6,236 verses)
  - **Passage (Meso)**: Groups of 19 consecutive verses
  - **Surah (Macro)**: Complete chapters (114 surahs)
- **Domain Bridging**: Maps modern concepts (e.g., "debugging", "stress") to Quranic themes
- **Contrastive Activation Addition (CAA)**: Difference-based steering vectors
- **Thematic Steering**: Steer toward specific themes (mercy, justice, patience, etc.)
- **Multiple Injection Modes**: `add`, `blend`, `replace`, and `clamp` (recommended for stability)
- **Comparison Mode**: Side-by-side comparison of steered vs baseline outputs
- **Native Reasoning Modes**: DeepSeek-R1, Qwen3, Phi-4 reasoning support
- **8 Supported LLMs**: From 135M to 3.8B parameters

### Based On

- [Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248)
- [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681)
- [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama) — clamp injection mode inspired by this work

## Architecture

```
machine-poi/
├── main.py                   # CLI entry point
├── compare_models.py         # Model comparison tool
├── config.py                 # Model configs, presets, SteeringDefaults
├── al-quran.txt              # Quran text (Arabic, ~739KB)
├── PAPER.md                  # Academic paper describing the methodology
├── requirements.txt          # Dependencies
├── Makefile                  # Test commands
├── src/                      # Core library (5 modules)
│   ├── steerer.py            # High-level QuranSteerer API
│   ├── llm_wrapper.py        # LLM hooks and steering injection
│   ├── quran_embeddings.py   # Text embedding & chunking
│   ├── steering_vectors.py   # Vector projection utilities
│   └── knowledge_base.py     # ChromaDB for MRA mode
├── experiments/              # Paper reproduction scripts
│   └── reproduce_paper.py    # Reproduces paper experiments
├── tests/                    # Comprehensive test suite (100 tests)
├── vectors/                  # Cached steering vectors
└── quran_db/                 # ChromaDB storage for MRA
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| `QuranSteerer` | High-level API orchestrating all components |
| `ContrastiveQuranSteerer` | CAA-based steering with positive/negative examples |
| `SteeredLLM` | Wraps HuggingFace models with activation hooks |
| `QuranEmbeddings` | Creates embeddings with LRU cache |
| `SteeringVectorExtractor` | Projects embeddings → LLM hidden dimensions |
| `QuranKnowledgeBase` | Multi-resolution ChromaDB for MRA retrieval |

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
from src import QuranSteerer, ContrastiveQuranSteerer

# Initialize
steerer = QuranSteerer(
    llm_model="deepseek-r1-1.5b",
    embedding_model="paraphrase-minilm",  # lightweight default
)

# Load models
steerer.load_models()

# Option 1: Standard Quran steering (uses Mean Activation)
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

# Contrastive steering (CAA)
caa_steerer = ContrastiveQuranSteerer(llm_model="deepseek-r1-1.5b")
caa_steerer.load_models()
caa_steerer.prepare_quran_contrastive()  # Quran vs neutral text
```

### Command Line

```bash
# Basic usage (uses deepseek-r1-1.5b by default)
python3 main.py --interactive

# With Quran Persona mode (recommended)
python3 main.py --quran-persona --interactive

# Use clamp injection for more stable steering at higher coefficients
python3 main.py --quran-persona --injection-mode clamp --coefficient 0.8 --interactive

# Compare steered vs baseline on test prompts
python3 main.py --compare

# Thematic steering toward specific concepts
python3 main.py --theme mercy --prompt "How should we treat others?"

# Multi-Resolution Analysis mode (requires --init-db first)
python3 main.py --init-db          # Build ChromaDB knowledge base
python3 main.py --mra --prompt "How should I deal with stress?"

# Enable native reasoning mode (uses model-specific config)
python3 main.py --llm deepseek-r1-1.5b --reasoning --prompt "What is wisdom?"

# Compare multiple models with the same prompt
./compare_models.py --models qwen3-0.6b deepseek-r1-1.5b --reasoning
./compare_models.py --list-models
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--llm MODEL` | LLM to steer (default: `deepseek-r1-1.5b`) |
| `--embedding MODEL` | Embedding model (default: `paraphrase-minilm`) |
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
| `deepseek-r1-1.5b` | 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ✅ `<think>` blocks |
| `qwen3-0.6b` | 0.6B | `Qwen/Qwen3-0.6B` | ✅ `enable_thinking` |
| `qwen2.5-0.5b` | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | — |
| `smollm2-135m` | 135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | — |
| `smollm2-360m` | 360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | — |
| `smollm3` | 3B | `HuggingFaceTB/SmolLM3-3B` | — |
| `gemma-270m` | 270M | `google/gemma-3-270m-it` | — |
| `phi4-mini` | 3.8B | `microsoft/Phi-4-mini-reasoning` | ✅ Math reasoning |

### Embedding Models

| Model | Dim | Arabic Support | Memory |
|-------|-----|----------------|--------|
| `paraphrase-minilm` | 384 | ✅ | ~0.5 GB |
| `bge-m3` | 1024 | ✅ | ~2.5 GB |
| `multilingual-e5` | 1024 | ✅ | ~2.0 GB |
| `qwen-embedding` | 3584 | ✅ | ~15 GB |

## How It Works

### 1. Data Preparation

The Quran text is loaded and chunked at three resolutions:
- **Verse**: Individual ayat (~6,236 verses)
- **Passage**: Groups of 19 consecutive verses
- **Surah**: Complete chapters using standard 114 Surah verse counts

### 2. Steering Vector Calculation (Mean Activation)

To steer the model, we do not simply project embeddings. Instead, we:
1. Sample a representative set of Quran verses.
2. Feed these verses into the LLM.
3. Extract the internal hidden states (activations) at each layer.
4. Compute the **mean activation vector** for each layer.

This vector represents the "direction" of Quranic content in the model's own latents.

### 3. Activation Injection

During inference, this mean activation vector is added (or clamped) to the model's current activations, "nudging" the generation towards the Quranic style and semantic space.

```python
llm.register_steering_hook(
    layer_idx=12,
    steering_vector=mean_activation_vector,
    coefficient=0.5,
    injection_mode="clamp"
)
```

### 4. Quran Persona Mode

Aggregates mean activations from Verse, Paragraph, and Surah levels with configurable weights:
- Verse: 50% (precise semantic signals)
- Paragraph: 35% (thematic context)
- Surah: 15% (foundational principles)

### 5. Domain Bridging

Maps user concepts (e.g., "bug", "deadline", "stress") to Quranic themes for better retrieval in MRA mode. This enables the system to find relevant Quranic guidance even for modern/technical topics.

| User Domain | Quranic Themes |
|-------------|----------------|
| debugging | patience, careful examination, seeking truth |
| stress | sabr, trust in Allah, peace of heart |
| teamwork | unity, brotherhood, cooperation, ummah |
| leadership | responsibility, trust, justice, consultation |

## Injection Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `add` | Add steering vector to activations | Gentle steering, lower coefficients |
| `blend` | Interpolate between original and steering | Balanced control |
| `replace` | Replace activations entirely | Maximum effect (use carefully) |
| `clamp` | Remove projection, add controlled amount | **Recommended for strong steering** |

The `clamp` mode is inspired by [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama). It first removes the existing projection of activations onto the steering direction, then adds a controlled amount. This prevents over-biasing and produces more fluent outputs at higher coefficients.

## Reproducing Paper Experiments

See [PAPER.md](PAPER.md) for the full academic paper. To reproduce experiments:

```bash
# Run all paper experiments
python3 experiments/reproduce_paper.py

# Run specific sections
python3 experiments/reproduce_paper.py --section 5.1  # Qualitative comparison
python3 experiments/reproduce_paper.py --section 5.2  # Thematic analysis
python3 experiments/reproduce_paper.py --section 5.3  # Coefficient sensitivity

# Quick test with fewer prompts
python3 experiments/reproduce_paper.py --quick
```

## Testing

The project includes a comprehensive test suite with 100 tests covering all modules:

```bash
# Run fast tests (excludes slow/integration)
make test

# Run all tests including slow ones
make test-all

# Run with coverage report
make test-cov

# Run specific test file
make test-file FILE=tests/test_steerer.py

# Run tests matching a pattern
make test-match MATCH="injection"
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_knowledge_base.py` | ChromaDB indexing/querying |
| `test_llm_wrapper.py` | Activation hooks, injection modes |
| `test_quran_embeddings.py` | Text chunking, embedding creation |
| `test_steerer.py` | End-to-end steering workflows |
| `test_steering_vectors.py` | Vector projection, contrastive methods |

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
