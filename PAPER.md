# Machine-POI: Multi-Resolution Activation Engineering for Semantic Steering via Quranic Embeddings

---

## Abstract

This paper presents **Machine-POI**, a framework for steering Large Language Models (LLMs) using activation engineering techniques derived from Quranic text. We introduce a novel **Multi-Resolution Analysis (MRA)** approach that extracts and injects semantic "Points of Intervention" (POIs) at three hierarchical levels: verse (micro), passage (meso), and surah (macro). By applying **Mean Activation Steering** and **Contrastive Activation Addition (CAA)**, Machine-POI enables precise thematic alignment of LLM responses with specific moral and cultural frameworks—without the computational overhead of fine-tuning.

Our system further employs a "Domain Bridging" heuristic to translate diverse user queries into relevant Quranic themes, facilitating cross-domain analogical reasoning. Experimental results demonstrate that Machine-POI successfully infuses Quranic moral reasoning into LLM outputs while preserving the base model's general capabilities.

**Keywords**: Activation Engineering, LLM Alignment, Semantic Steering, Multi-Resolution Analysis, Quranic NLP

---

## 1. Introduction

### 1.1 Motivation
The alignment of Large Language Models (LLMs) with specific human values, cultural contexts, or religious frameworks is a critical challenge in modern AI safety research. Traditional approaches rely on Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF). While effective, these methods are:

1. **Resource-intensive**: Requiring large datasets and significant compute.
2. **Prone to catastrophic forgetting**: The model may lose general capabilities.
3. **Inflexible**: Difficult to dynamically adjust alignment at inference time.

### 1.2 Our Approach
Machine-POI explores an alternative paradigm: **Activation Engineering**. Instead of modifying model weights, we manipulate the model's internal hidden states (activations) during inference. By injecting steering vectors derived from the Quran, we guide the LLM's reasoning and tone toward specific thematic "Points of Intervention" (POIs).

### 1.3 Contributions
1. **Multi-Resolution Analysis (MRA)**: A hierarchical framework for extracting semantic signals at verse, passage, and surah levels.
2. **Domain Bridging**: A heuristic mapping system that translates modern concepts to Quranic themes.
3. **Quran Persona**: A unified steering profile aggregating multi-resolution activations.
4. **Open-source implementation**: A modular Python framework compatible with HuggingFace Transformers.

---

## 2. Related Work

### 2.1 Activation Addition (ActAdd)
Turner et al. (2024) demonstrated that specific semantic directions in the hidden space of an LLM can be isolated and added to the residual stream to influence the model's output. Their work showed that simple vector arithmetic in activation space can produce reliable behavioral changes.

### 2.2 Contrastive Activation Addition (CAA)
Rimsky et al. (2024) introduced CAA to enhance steering precision by computing the difference between activations of positive examples (e.g., a specific virtue) and negative examples (e.g., neutral or opposing text). This approach provides finer control over steering direction.

### 2.3 Retrieval-Augmented Generation (RAG)
Lewis et al. (2020) proposed RAG as a method for grounding LLM outputs in external knowledge. While RAG operates at the input level (prompt augmentation), Machine-POI operates at the hidden state level, providing a complementary and potentially more robust form of semantic injection.

---

## 3. Methodology

### 3.1 Multi-Resolution Analysis (MRA)
A core contribution of Machine-POI is the hierarchical representation of semantic context. The Quran is structured into 114 Surahs (chapters) containing 6,236 verses (ayat). We leverage this natural hierarchy:

| Resolution | Granularity | Semantic Signal | Use Case |
|------------|-------------|-----------------|----------|
| **Micro (Verse)** | Individual ayat | Precise, specific concepts | Targeted thematic steering |
| **Meso (Passage)** | 5 consecutive verses | Contextual themes | Balanced steering |
| **Macro (Surah)** | Entire chapter | Foundational principles | Holistic persona |

The final Quran Persona vector is computed as a weighted combination:

$$\vec{v}_{persona} = w_{verse} \cdot \vec{v}_{verse} + w_{passage} \cdot \vec{v}_{passage} + w_{surah} \cdot \vec{v}_{surah}$$

where default weights are $w_{verse} = 0.5$, $w_{passage} = 0.35$, $w_{surah} = 0.15$.

### 3.2 Domain Bridging
To facilitate interaction between modern user queries and classical texts, we implement a **Domain Bridge Map**. This heuristic aligns contemporary concepts with Quranic themes:

| User Domain | Quranic Themes |
|-------------|----------------|
| debugging | patience, careful examination, seeking truth |
| stress | sabr, trust in Allah, peace of heart |
| teamwork | unity, brotherhood, cooperation, ummah |
| failure | perseverance, learning, hope, never despair |
| leadership | responsibility, trust, justice, consultation |

### 3.3 Vector Extraction and Injection
The steering vector $\vec{v}_l$ for layer $l$ is computed as the mean normalized activation:

$$\vec{v}_l = \text{normalize}\left(\frac{1}{N} \sum_{i=1}^{N} \text{activation}_l(T_i)\right)$$

where $T_i$ represents the tokens of the Quranic text sample. During inference, this vector modifies the hidden state $h_l$:

$$h'_l = h_l + \alpha \cdot \vec{v}_l$$

We support multiple injection modes:
- **add**: Direct vector addition (default)
- **blend**: Linear interpolation with ratio $\beta$: $h'_l = (1-\beta) \cdot h_l + \beta \cdot \vec{v}_l$
- **clamp**: Normalized addition to prevent saturation

---

## 4. System Architecture

Machine-POI is implemented as a modular Python framework with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                      QuranSteerer                           │
│  (Orchestrator: Domain Bridging, MRA, Generation)           │
├─────────────────┬─────────────────┬─────────────────────────┤
│   SteeredLLM    │ QuranEmbeddings │  QuranKnowledgeBase     │
│ (Hook-based     │ (Embedding +    │  (ChromaDB + Multi-     │
│  Activation     │  LRU Cache)     │   Resolution Index)     │
│  Injection)     │                 │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 4.1 SteeredLLM
A non-invasive wrapper for HuggingFace Transformers that registers forward hooks at residual stream locations (typically `post_attention_layernorm`). Supports:
- 8-bit and 4-bit quantization via `bitsandbytes`
- Native reasoning modes for DeepSeek-R1, Qwen3, and Phi-4

### 4.2 QuranKnowledgeBase
A vector database built on ChromaDB that maintains three collections (verse, passage, surah) with cosine similarity search. Supports:
- Multi-resolution retrieval
- Domain bridge query expansion

---

## 5. Experimental Results

We evaluated Machine-POI using **Qwen3-0.6B** as the base model, with **BGE-M3** for embeddings. All experiments were conducted on a single NVIDIA RTX GPU.

### 5.1 Qualitative Comparison

**Prompt**: *"What is the meaning of life?"*

| Condition | Output |
|-----------|--------|
| **Unsteered** | "The meaning of life is a philosophical question that has been debated for centuries. Common answers include happiness, success, or personal fulfillment..." |
| **Steered (Quran Persona)** | "The meaning of life, in its deepest essence, is to worship and serve the One who created us, to live with gratitude and patience, and to prepare for the eternal journey ahead. True purpose comes from submission to the divine will and building a legacy of righteousness..." |

**Prompt**: *"How should I deal with a bug in my code?"*

| Condition | Output |
|-----------|--------|
| **Unsteered** | "Debug systematically: check logs, add print statements, use a debugger, isolate the issue..." |
| **Steered + MRA** | "Approach this with patience (*sabr*) and careful examination. Just as truth is sought with diligence, so too should we methodically trace the source of error. Begin with gratitude that the problem has revealed itself, then proceed with disciplined reasoning..." |

### 5.2 Thematic Consistency Analysis

We tested steering effectiveness across 10 prompts with the "mercy" theme:

| Metric | Unsteered | Steered (α=0.5) |
|--------|-----------|-----------------|
| Mercy keyword frequency | 0.3 / response | 2.1 / response |
| Religious reference rate | 5% | 68% |
| Response coherence (1-5) | 4.2 | 4.0 |

### 5.3 Coefficient Sensitivity

| Coefficient (α) | Thematic Strength | Coherence | Notes |
|-----------------|-------------------|-----------|-------|
| 0.2 | Subtle | High | Gentle influence |
| 0.5 | Moderate | High | Recommended default |
| 0.8 | Strong | Medium | Noticeable thematic shift |
| 1.2 | Very Strong | Lower | Risk of repetition |

---

## 6. Discussion

### 6.1 Findings
1. **Effective thematic steering**: Machine-POI successfully infuses Quranic themes into LLM outputs without fine-tuning.
2. **Preserved reasoning**: Base model capabilities remain largely intact, especially with moderate coefficients.
3. **Domain bridging utility**: The heuristic mapping enables meaningful cross-domain analogies.

### 6.2 Limitations
1. **Language dependency**: Current implementation focuses on Arabic Quranic text; translation quality may affect results.
2. **Static domain bridges**: The keyword map is manually curated; future work could learn bridges from data.
3. **Coefficient tuning**: Optimal steering strength varies by model and task.

---

## 7. Future Work

1. **Dynamic Layer Selection**: Automatically identifying optimal intervention points based on query complexity.
2. **Learned Domain Bridges**: Using embedding similarity to generate bridges automatically.
3. **Multi-cultural Extension**: Applying the framework to other religious or philosophical texts.
4. **Quantitative Evaluation**: Developing standardized benchmarks for semantic steering.

---

## 8. Conclusion

Machine-POI demonstrates that activation engineering provides a powerful, flexible approach to aligning LLM outputs with specific cultural and moral frameworks. By combining Multi-Resolution Analysis with Domain Bridging, we enable nuanced semantic steering that preserves base model capabilities while introducing targeted thematic influences. Our open-source implementation provides a foundation for future research in culturally-aware AI systems.

---

## Appendix A: Reproducing Experiments

All experiments in this paper can be reproduced using the provided script:

```bash
# Clone the repository
git clone https://github.com/gutama/machine-poi.git
cd machine-poi

# Install dependencies
pip install -r requirements.txt

# Run all experiments
python experiments/reproduce_paper.py --model qwen3-0.6b

# Run specific sections
python experiments/reproduce_paper.py --section 5.1  # Qualitative comparison
python experiments/reproduce_paper.py --section 5.2  # Thematic analysis
python experiments/reproduce_paper.py --section 5.3  # Coefficient sensitivity
python experiments/reproduce_paper.py --section mra  # MRA mode demo

# Quick test (fewer prompts)
python experiments/reproduce_paper.py --quick
```

**Hardware Requirements**:
- GPU: NVIDIA GPU with ≥8GB VRAM (recommended)
- CPU-only: Supported but significantly slower
- RAM: ≥16GB

---

## References

Chroma Team. (2023). *Chroma: The open source embedding database*. https://www.trychroma.com/

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024). Steering Llama 2 via Contrastive Activation Addition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15504–15522). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.828/

Turner, A. M., Thiergart, L., Leech, G., Udell, D., Mini, U., & MacDiarmid, M. (2024). *Activation Addition: Steering Language Models Without Optimization*. arXiv.org. https://doi.org/10.48550/arXiv.2308.10248
