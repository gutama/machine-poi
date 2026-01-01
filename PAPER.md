# Machine-POI: Multi-Resolution Activation Engineering for Semantic Steering via Quranic Embeddings

**Abstract**
This paper presents Machine-POI, a framework for steering Large Language Models (LLMs) using activation engineering techniques derived from Quranic text. We introduce a novel Multi-Resolution Analysis (MRA) approach that extracts and injects semantic "Points of Intervention" (POIs) at three levels: verse (micro), passage (meso), and surah (macro). By applying Mean Activation Steering and Contrastive Activation Addition (CAA), Machine-POI enables precise thematic alignment of LLM responses with specific moral and cultural frameworks without the need for computational-heavy fine-tuning. Our system further employs a "Domain Bridging" heuristic to translate diverse user queries into relevant Quranic themes, facilitating cross-domain analogical reasoning.

---

## 1. Introduction
The alignment of Large Language Models (LLMs) with specific human values, cultural contexts, or religious frameworks is typically achieved through Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF). While effective, these methods are resource-intensive and often lead to "catastrophic forgetting" or the dilution of the model's original capabilities.

Machine-POI explores an alternative paradigm: **Activation Engineering**. Instead of modifying model weights, we manipulate the model's internal hidden states (activations) during inference. By injecting steering vectors derived from the Quran, we guide the LLM's reasoning and tone toward specific thematic "Points of Intervention."

## 2. Theoretical Background
### 2.1 Activation Addition (ActAdd)
Turner et al. (2024) demonstrated that specific semantic directions in the hidden space of an LLM can be isolated and added to the residual stream to influence the model's output. Machine-POI builds on this by using **Mean Activation Steering**, where the steering vector is the average activation of a representative set of tokens from a target domain.

### 2.2 Contrastive Activation Addition (CAA)
Rimsky et al. (2024) introduced CAA to enhance steering precision by computing the difference between activations of a positive example (e.g., a specific virtue) and a negative example (e.g., neutral or opposing text). Machine-POI implements `ContrastiveQuranSteerer` to allow for high-fidelity thematic steering.

## 3. Methodology
### 3.1 Multi-Resolution Analysis (MRA)
A core contribution of Machine-POI is the hierarchical representation of semantic context. Steering vectors are extracted at three resolutions:
1.  **Micro (Verse)**: Precise semantic signals from individual verses.
2.  **Meso (Passage)**: Thematic context from grouped verses (paragraphs).
3.  **Macro (Surah)**: Foundational principles derived from entire chapters.

### 3.2 Domain Bridging
To facilitate interaction between modern user queries and traditional texts, we implement a **Domain Bridge Map**. This heuristic aligns technical or contemporary concepts (e.g., "debugging," "stress," "teamwork") with relevant Quranic themes (e.g., "patience," "tranquility," "shura"). These bridges serve as retrieval queries for the MRA engine.

### 3.3 Vector Extraction and Injection
The steering vector $\vec{v}_l$ for layer $l$ is computed as the mean normalized activation:
$$\vec{v}_l = \text{unit}(\frac{1}{N} \sum_{i=sample} \text{act}_l(T_i))$$
where $T_i$ represents the tokens of the Quranic text sample. During inference, this vector is added to the hidden state $h_l$ with a coefficient $\alpha$:
$$h'_l = h_l + \alpha \cdot \vec{v}_l$$

## 4. System Architecture
Machine-POI is implemented as a modular Python framework:
- **`SteeredLLM`**: A non-invasive wrapper for HuggingFace Transformers that registers forward hooks at residual stream locations (e.g., `post_attention_layernorm`).
- **`QuranEmbeddings`**: Manages semantic vectorization and includes an LRU cache for computational efficiency.
- **`QuranKnowledgeBase`**: A vector database built on ChromaDB (Chroma Team, 2023) that supports multi-resolution retrieval.
- **`QuranSteerer`**: The central orchestrator that manages domain bridging, dynamic steering, and MRA prompt construction.

## 5. Implementation Details
The framework supports various injection modes:
- **`add`**: Simple vector addition.
- **`blend`**: Linear interpolation between the original activation and the steering vector.
- **`clamp`**: Normalizes the activation to prevent saturation during strong steering.

To handle memory constraints, we implement aggressive garbage collection and CUDA cache clearing, alongside a multi-level caching system for steering vectors.

## 6. Discussion and Future Work
Experimental results with models such as DeepSeek-R1 and Qwen3 indicate that Machine-POI successfully shifts the model's internal hierarchy of relevance toward Quranic principles while maintaining the base model's reasoning capabilities. 

Future research will focus on:
- **Dynamic Layer Selection**: Automatically identifying optimal intervention points based on query complexity.
- **Online Learning for POIs**: Refining steering vectors based on human feedback loop.

## 7. References
Chroma Team. (2023). *Chroma: The open source embedding database*. https://www.trychroma.com/

Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024). Steering Llama 2 via Contrastive Activation Addition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 15504–15522). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.828/

Turner, A. M., Thiergart, L., Leech, G., Udell, D., Mini, U., & MacDiarmid, M. (2024). *Activation Addition: Steering Language Models Without Optimization*. arXiv.org. https://doi.org/10.48550/arXiv.2308.10248
