# Global Workspace Improvement Plan for Machine-POI

This plan adapts ideas from Anthropic's 2026 Global Workspace research and the related Transformer Circuits workspace write-up for Machine-POI's Quranic activation-steering architecture.

## Concepts to incorporate

- **Workspace-like representations**: target the intermediate model states most likely to be reusable by downstream reasoning rather than only early parsing or late token-output states.
- **Selectivity**: prefer narrow, interpretable interventions over broad perturbations that affect every layer equally.
- **Verbalizable concepts**: connect steering vectors to explicit Quranic concepts, themes, and bridge terms so users can inspect what the system is attempting to steer toward.
- **Diagnostics before trust**: report activation norms, cosine alignment, and perturbation size so steering can be audited rather than treated as a black box.
- **Counterfactual reflection as an experiment**: keep reflection-style interventions in experiments until they are validated and documented.

## Execution plan

### 1. Add workspace-aware layer selection

Machine-POI should offer a `workspace` layer distribution that emphasizes intermediate layers where reusable internal representations are most likely to live. This avoids treating all layers as equally suitable intervention points.

**Status:** Implemented as `select_workspace_layers()` and `layer_distribution_scale()` in `src/steerer.py`, with support in both static and dynamic steering application paths.

### 2. Add workspace diagnostics

Steering should expose lightweight metrics that make internal perturbations inspectable:

- activation norm,
- steering-vector norm,
- mean cosine similarity,
- mean projection magnitude,
- relative perturbation size.

**Status:** Implemented as `src/workspace_diagnostics.py` with tensor-only unit tests. Runtime integrations can call `SteeredLLM.get_steering_diagnostics()` after generation has captured hook activations.

### 3. Prefer activation-derived steering vectors

The default Quran Persona and Quran steering paths should continue to rely on mean activations extracted from the steered LLM, rather than uncalibrated random projection from embedding space. Projection-based utilities should be treated as experimental unless calibrated.

**Status:** Documented as an architecture direction. Future implementation should add runtime warnings to projection-based steering paths.

### 4. Add a workspace audit mode

A future CLI mode should compare baseline and steered generations across prompts and report religious-reference intensity, refusal changes, length shifts, and diagnostics.

**Status:** Planned. The diagnostics module provides a foundation for this mode.

### 5. Add counterfactual reflection experiments

Counterfactual reflection should be implemented in `experiments/`, not as default runtime behavior. The experiment should compare normal prompts with interrupted reflection prompts and build contrastive vectors from the differences.

**Status:** Planned.

### 6. Add a structured concept vocabulary

Move Quranic bridge themes into structured concept objects containing canonical names, aliases, Arabic/Islamic terms, related themes, and safety notes.

**Status:** Planned.

### 7. Add oversteering safeguards

Future work should cap relative perturbation size per layer and surface warnings when coefficients cause broad hidden-state shifts.

**Status:** Planned. The new diagnostics expose the relative perturbation metric needed for this safeguard.

### 8. Clarify documentation

The README should distinguish retrieval grounding, activation intervention, CAA, and workspace-style interpretability.

**Status:** Implemented with a workspace-inspired roadmap section.
