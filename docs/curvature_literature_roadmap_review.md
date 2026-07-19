# Curvature & Manifold Literature Review — Relevance to Machine-POI's Roadmap

**Purpose:** This document maps recent (2025–2026) literature on curvature, manifolds, and geometric attention onto Machine-POI's existing roadmap — specifically the [Global Workspace Improvement Plan](global_workspace_improvement_plan.md) and the attention-transport diagnostics already implemented in `src/workspace_diagnostics.py` and `experiments/gpt_on_manifolds_v4.py`.

---

## 1. Where the project already stands

Machine-POI's roadmap has two geometry-adjacent workstreams already **implemented**:

| Roadmap item | Status | File |
|---|---|---|
| Workspace-aware layer selection | Implemented | `src/steerer.py` (`select_workspace_layers()`) |
| Pointwise steering diagnostics (norm, cosine, projection) | Implemented | `src/workspace_diagnostics.py` |
| **Attention-transport (curvature) diagnostics** — non-abelian ratio ρ, holonomy, discrete Cartan curvature Ω = dω + ω∧ω | Implemented | `src/workspace_diagnostics.py`, `experiments/gpt_on_manifolds_v4.py`, `experiments/steered_vs_baseline_transport.py` |

The curvature diagnostics treat each attention head as a **discrete connection** on a sequence fiber bundle: attention weights define a connection bivector ω_t, and its variation (dω) vs. commutator (ω∧ω) terms separate "position-dependent but commutative" heads from "genuinely order-sensitive" heads. `gpt_on_manifolds_v4.py` additionally trains a toy GPT with embeddings projected onto **hyperbolic (Poincaré), spherical, product H×S, and Grassmannian manifolds**, using a Riemannian natural-gradient optimizer.

Planned-but-not-implemented roadmap items that geometric literature can inform:
- Workspace audit CLI mode
- Oversteering safeguards (capping relative perturbation)
- Structured Quranic concept vocabulary
- Counterfactual reflection experiments

---

## 2. Paper-by-paper relevance mapping

### 2.1 Attention-as-connection / curvature diagnostics (directly extends existing work)

| Paper | Core idea | Relevance to roadmap |
|---|---|---|
| [The Curved Spacetime of Transformer Architectures](https://arxiv.org/abs/2511.03060) (Nov 2025) | Attention is a discrete connection transporting value vectors on a curved semantic manifold; proposes turning-angle and length-to-chord-ratio curvature diagnostics, plus a "deflection" test analogous to gravitational lensing. | Provides an **independent, published formalization** of the same idea Machine-POI already implements (attention-as-transport). Its turning-angle / length-to-chord diagnostics are simpler than the bivector/holonomy machinery in `gpt_on_manifolds_v4.py` and could serve as a **lightweight cross-check** in the planned audit CLI mode — cheap enough to run per-prompt without the O(n³) triangular-loop cost of holonomy. The "deflection under controlled context edits" experiment is a direct template for testing whether Quran steering *bends* representation trajectories in a meaning-consistent way, which is exactly what `steered_vs_baseline_transport.py` tries to detect via ρ and holonomy deltas.|
| [RiemannFormer: A Framework for Attention in Curved Spaces](https://arxiv.org/abs/2506.07405) (Jun 2025) | Reformulates Q·K attention as parallel transport between tangent spaces under a learned Riemannian metric M_i at each token position; requires transporting keys into the query's tangent frame before the inner product. | Gives a **principled alternative attention formula** (not just a diagnostic) that Machine-POI could adopt if it ever wants to bake curvature *into* the steered model's forward pass rather than only measuring it post hoc. Also useful as a sanity check: RiemannFormer's parallel-transport operator is structurally the same object as the `transport_map()` (T_t = exp(−ηω_t)) already in `gpt_on_manifolds_v4.py`. |
| [Gating Enables Curvature: A Geometric Expressivity Gap in Attention](https://arxiv.org/pdf/2604.14702.pdf) (Apr 2026) | Proves ungated attention is restricted to intrinsically **flat** statistical manifolds (its outputs are affine combinations of values); multiplicative gating is required to reach non-flat/positively-curved geometries. | Directly relevant to interpreting the project's own "flat / commutative-varying / order-sensitive" per-head taxonomy (§6.4 of `gpt_on_manifolds_v4.py`). If a head is architecturally ungated, this paper predicts it *must* register as "FLAT" or low-ρ under the diagnostics — a testable hypothesis the roadmap's audit mode could report on (e.g., flag heads/architectures where flatness is structural rather than learned). |
| [The Bayesian Geometry of Transformer Attention](https://arxiv.org/abs/2512.22471) (Dec 2025) | Shows attention implements content-addressable routing of a Bayesian belief state carried in the residual stream. | Less a curvature paper than a routing-semantics paper, but complements the roadmap's framing that steering should be evaluated on whether it changes *how context is routed*, not just *where representations sit*. Could motivate a belief-tracking diagnostic alongside ρ/holonomy in the planned audit mode. |

### 2.2 Hyperbolic / mixed-curvature LLM architectures (relevant to the manifold *training* experiment, not the steering pipeline itself)

| Paper | Core idea | Relevance to roadmap |
|---|---|---|
| [Hyperbolic Large Language Models](https://arxiv.org/html/2509.05757v1) (Sep 2025, survey) | Taxonomy of HypLLMs: exp/log-map hybrids, hyperbolic fine-tuning, fully hyperbolic models, hyperbolic SSMs. | Useful **background reading** for anyone extending `gpt_on_manifolds_v4.py` beyond a toy testbed — the taxonomy clarifies which manifold-integration strategy (hybrid vs. fully hyperbolic) would be tractable to port into the main `SteeredLLM` wrapper. |
| [HELM: Hyperbolic LLMs via Mixture-of-Curvature Experts](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d1e2f808a51842eedaf6ef0099d716c6-Abstract-Conference.html) (NeurIPS 2025) | Billion-scale fully hyperbolic LLM; each expert operates in its own curvature; hyperbolic RoPE/RMSNorm/attention. | Not applicable to steering an *existing pretrained Euclidean* model (Machine-POI's actual use case), since HELM requires training from scratch in hyperbolic space. Relevant only if the roadmap's "Future Work" ever pivots to training a small Quran-native model rather than steering off-the-shelf checkpoints. |
| [CAT: Curvature-Adaptive Transformers](https://arxiv.org/abs/2510.01634) (Oct 2025) | Learns per-token routing across Euclidean/hyperbolic/spherical attention branches. | Conceptually validates the project's `MANIFOLD_TYPE` product-space experiment (H^8 × S^8) — CAT shows *mixed* geometry outperforms any single fixed geometry, supporting the choice to keep testing product manifolds rather than committing to pure hyperbolic. |
| [Curve Your Attention: Mixed-Curvature Transformers](http://arxiv.org/pdf/2309.04082.pdf) | Learnable per-head sectional curvature on a product-stereographic manifold. | Direct ancestor of the `MANIFOLD_TYPE='product'` path in `gpt_on_manifolds_v4.py`; the roadmap could cite this as the architectural precedent and consider making curvature itself learnable (currently `HYPERBOLIC_C` is a fixed constant). |
| [Hyperbolic Fine-tuning for LLMs (HypLoRA)](http://arxiv.org/pdf/2410.04010.pdf) | LoRA-style low-rank adaptation performed in hyperbolic space; gains on hierarchical reasoning tasks. | Most **directly actionable** hyperbolic paper for the roadmap: HypLoRA fine-tunes an *existing* pretrained LLM without full retraining — structurally analogous to Machine-POI's activation-steering approach (no full fine-tuning) but operating in a different intervention space (weight-space LoRA deltas vs. residual-stream activations). Item 3 of the roadmap ("prefer activation-derived steering vectors... projection-based utilities experimental") could reference HypLoRA as a possible complementary/alternative intervention to evaluate. |
| [Position: Foundation Models Should Embrace Non-Euclidean Geometries](https://arxiv.org/abs/2504.08896) (Apr 2025) | Argues Euclidean geometry is a scaling bottleneck; proposes a roadmap for non-Euclidean foundation models. | Broad motivational framing; supports treating `gpt_on_manifolds_v4.py` as more than a toy — i.e., justifies eventually scaling the manifold experiment past a 30-name toy dataset if resources allow. |

### 2.3 Riemannian structure of pretrained representations (interpretability-focused, closest to the *diagnostics* half of the roadmap)

| Paper | Core idea | Relevance to roadmap |
|---|---|---|
| [Riemannian Geometry for Pre-trained Language Model Embeddings](https://arxiv.org/html/2607.07047v1) (Jul 2026) | Pulls back a Riemannian metric from a PLM's Jacobian; aggregates token embeddings via Fréchet means on the SPD manifold ("Riemannian Mean Pooling"); outperforms Euclidean pooling on linguistically structured tasks. | Suggests a **new diagnostic**: instead of (or alongside) mean-activation steering vectors, compute a Fréchet-mean steering vector on the pulled-back metric. Could reduce noise in the Quran Persona vector, which currently uses a plain arithmetic mean of activations (README §2). Worth a follow-up experiment. |
| [Latent Semantic Manifolds in Large Language Models](https://arxiv.org/pdf/2603.22301.pdf) (Mar 2026) | Formalizes an LLM latent manifold with a Fisher-information metric; shows curvature spikes correlate with polysemy/semantic ambiguity and proposes curvature as a real-time training diagnostic. | Directly extends the roadmap's diagnostics philosophy ("diagnostics before trust"). Curvature-spike monitoring could become an **oversteering safeguard** (roadmap item 7): if steering pushes a layer's local curvature far outside its baseline range, flag it as a potential representation-collapse or instability risk, analogous to how the paper proposes detecting training instabilities. |
| [RiemannInfer](https://www.nature.com/articles/s41598-026-37328-x) (Nature Sci Reports, Jan 2026) | Builds a Riemannian manifold from attention distributions; uses geodesics/curvature for inference-path planning and interpretability. | Less about steering, more about efficiency — lower priority for Machine-POI, but its geodesic/curvature-based "reasoning path" visualization could inspire a visualization component for the planned workspace audit CLI mode (e.g., plotting the steered vs. baseline generation path through curvature space). |

---

## 3. Recommended roadmap updates

Based on the above mapping, three concrete additions to `docs/global_workspace_improvement_plan.md` are suggested:

1. **Item 2b (attention-transport diagnostics) — add a lightweight cross-check.**
   Implement the turning-angle / length-to-chord curvature diagnostic from *[The Curved Spacetime of Transformer Architectures](https://arxiv.org/abs/2511.03060)* as a cheaper companion to the existing bivector/holonomy diagnostics, since it avoids the O(n³) triangular-loop cost and can run per-token during interactive sessions.

2. **Item 7 (oversteering safeguards) — define a curvature-spike threshold.**
   Following *[Latent Semantic Manifolds in LLMs](https://arxiv.org/pdf/2603.22301.pdf)*, treat a sudden rise in local (PCA or sectional) curvature at a steered layer, relative to its unsteered baseline, as an early-warning signal for representation instability — complementing the existing relative-perturbation metric.

3. **New experimental item — Fréchet-mean steering vectors.**
   Following *[Riemannian Geometry for Pre-trained LM Embeddings](https://arxiv.org/html/2607.07047v1)*, add an experiment comparing the current arithmetic-mean Quran Persona vector against a Fréchet mean computed on the pulled-back Jacobian metric, to test whether it yields more stable or more thematically consistent steering at a given coefficient.

Lower-priority / background-only: the hyperbolic-LLM architecture papers (HELM, Hypformer, CAT, Curve Your Attention) are valuable context for the `gpt_on_manifolds_v4.py` testbed but are not directly actionable for the steering pipeline, since they assume training from scratch rather than steering an existing checkpoint. HypLoRA is the one exception worth a scoping note, as it shares Machine-POI's "no full fine-tuning" constraint.

---

## 4. Full reference list

- Turner, A. M. et al. (2024). *Activation Addition: Steering Language Models Without Optimization.* [arXiv:2308.10248](https://doi.org/10.48550/arXiv.2308.10248)
- Rimsky, N. et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.828/)
- *The Curved Spacetime of Transformer Architectures.* [arXiv:2511.03060](https://arxiv.org/abs/2511.03060)
- *RiemannFormer: A Framework for Attention in Curved Spaces.* [arXiv:2506.07405](https://arxiv.org/abs/2506.07405)
- *Gating Enables Curvature: A Geometric Expressivity Gap in Attention.* [arXiv:2604.14702](https://arxiv.org/pdf/2604.14702.pdf)
- *The Bayesian Geometry of Transformer Attention.* [arXiv:2512.22471](https://arxiv.org/abs/2512.22471)
- *Hyperbolic Large Language Models* (survey). [arXiv:2509.05757](https://arxiv.org/html/2509.05757v1)
- *HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts.* NeurIPS 2025. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d1e2f808a51842eedaf6ef0099d716c6-Abstract-Conference.html)
- *CAT: Curvature-Adaptive Transformers for Geometry-Aware Attention.* [arXiv:2510.01634](https://arxiv.org/abs/2510.01634)
- *Curve Your Attention: Mixed-Curvature Transformers.* [arXiv:2309.04082](http://arxiv.org/pdf/2309.04082.pdf)
- *Hyperbolic Fine-tuning for Large Language Models (HypLoRA).* [arXiv:2410.04010](http://arxiv.org/pdf/2410.04010.pdf)
- *Position: Foundation Models Should Embrace Non-Euclidean Geometries.* [arXiv:2504.08896](https://arxiv.org/abs/2504.08896)
- *Riemannian Geometry for Pre-trained Language Model Embeddings.* [arXiv:2607.07047](https://arxiv.org/html/2607.07047v1)
- *Latent Semantic Manifolds in Large Language Models.* [arXiv:2603.22301](https://arxiv.org/pdf/2603.22301.pdf)
- *RiemannInfer: Improving Transformer Inference through Riemannian Geometry.* Nature Scientific Reports (2026). [nature.com](https://www.nature.com/articles/s41598-026-37328-x)
- Nickel, M. & Kiela, D. (2017). *Poincaré Embeddings for Learning Hierarchies.*
- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning.*
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds.*
- Hestenes, D. & Sobczyk, G. (1984). *Clifford Algebra to Geometric Calculus.*
- Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

**Note:** The repo's internal references ("Is Attention Commutative? Quantifying Contextuality via a Discrete Cartan Curvature Diagnostic," Feb 2026) could not be located as a distinct indexed publication during this review. Its equations align closely with classical Cartan-geometry/Ehresmann-connection formalism and with the independently published papers listed in §2.1 above, which can serve as citable substitutes or cross-checks if the internal reference needs external validation.
