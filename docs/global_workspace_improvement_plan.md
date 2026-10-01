# Global Workspace Improvement Plan for Machine-POI

This plan adapts ideas from Anthropic's 2026 Global Workspace research and the related Transformer Circuits workspace write-up for Machine-POI's Quranic activation-steering architecture.

Implementation status reviewed 2026-09-29. This is a research roadmap. The
[guardian architecture](architecture.md) enforces structured tool grants
independently of steering and geometry; none of the diagnostics below grants
authority. See the [steering guide](steering_guide.md) for current APIs and
[testing guide](testing.md) for the distinction between runtime checks and
behavioral evidence.

## Current priority and geometry interpretation (2026-09-30)

The [containment/geometry research design](containment_geometry_research.md)
sets the current experimental order: host/agent baseline, bounded low-rank rotors,
then learned metrics and shadow signals only if incremental benefit is measured.
Workspace layer choice remains an ablation, not a proven safety mechanism.

The implemented transport algorithm supplies heuristic generator statistics.
Its adjacent differences plus commutators do not derive a curvature two-form;
`T_k T_j T_i` is not a closed loop of directed edge transports. Earlier “Cartan
curvature” and “holonomy” terminology below is historical. The new design
specifies frames, inverse links, pure-gauge and constant-generator controls, and
versioned migration before claiming genuine curvature. Run
`python experiments/geometry_sanity.py` for the mathematical fixtures.

## Concepts to incorporate

- **Workspace-like representations**: target the intermediate model states most likely to be reusable by downstream reasoning rather than only early parsing or late token-output states.
- **Selectivity**: prefer narrow, interpretable interventions over broad perturbations that affect every layer equally.
- **Verbalizable concepts**: connect steering vectors to explicit Quranic concepts, themes, and bridge terms so users can inspect what the system is attempting to steer toward.
- **Diagnostics before trust**: report activation norms, cosine alignment, and perturbation size so steering can be audited rather than treated as a black box.
- **Counterfactual reflection as an experiment**: keep reflection-style interventions in experiments until they are validated and documented.

## Execution plan

### 1. Add workspace-aware layer selection

Machine-POI should offer a `workspace` layer distribution that emphasizes intermediate layers where reusable internal representations are most likely to live. This avoids treating all layers as equally suitable intervention points.

**Status:** Implemented as `select_workspace_layers()` and `layer_distribution_scale()` in `machine_poi/steerer.py`, with support in both static and dynamic steering application paths.

### 2. Add workspace diagnostics

Steering should expose lightweight metrics that make internal perturbations inspectable:

- activation norm,
- steering-vector norm,
- mean cosine similarity,
- mean projection magnitude,
- relative perturbation size.

**Status:** Implemented as `machine_poi/workspace_diagnostics.py` with tensor-only unit tests. Relative perturbation now uses the actual update for add, blend, replace and clamp. After high-level `generate` or `generate_with_graph`, read `QuranSteerer.last_run_diagnostics`: session restoration clears captured tensors. Low-level callers can use `SteeredLLM.get_steering_diagnostics()` while their enabled hooks still hold the captured activations. These summaries do not certify behavior or permissions.

### 2b. Add attention-transport (curvature) diagnostics

**Interpretation correction:** this section records the legacy heuristic design;
see the current geometry contract above before interpreting its field names.

Beyond pointwise perturbation metrics, steering should be auditable for whether it changes *how* the model routes context, not just where representations sit. Following the discrete Cartan curvature framework of "Is Attention Commutative?" (2026), each attention head's transport geometry is summarized by:

- the non-abelian ratio ρ = Σ‖ω∧ω‖ / (Σ‖dω‖ + Σ‖ω∧ω‖), measuring how much of a head's curvature comes from non-commuting local transport generators (order sensitivity / path dependence),
- holonomy angles from exact transport maps T_t = exp(−η ω_t) around triangular position loops (loop-induced rotation, the "angle of context"),
- variation vs. commutator energy, distinguishing position-dependent-but-commutative heads from genuinely order-sensitive heads.

Comparing these per-head profiles with steering enabled vs. disabled measures changes in the constructed transport diagnostic. Interpreting such a change as useful context routing requires output/task controls: the [committed results](../experiments/results/README.md) include collapse and persona spillover. A standalone manifold/transport testbed lives in `experiments/gpt_on_manifolds_v4.py`; its presence does not independently validate the cited research interpretation.

**Status:** Implemented as `connection_bivectors()`, `summarize_attention_transport()`, `summarize_attention_transport_heads()`, and `pooled_non_abelian_ratio()` in `machine_poi/workspace_diagnostics.py` with tensor-only unit tests. Runtime integration: `SteeredLLM.get_attention_transport_diagnostics(prompt)` captures per-head attention weights and query/value projections in one forward pass (grouped-query attention supported); wrap it in `steering_disabled()` to obtain the unsteered baseline.

### 3. Prefer activation-derived steering vectors

The default Quran Persona and Quran steering paths should continue to rely on mean activations extracted from the steered LLM, rather than uncalibrated random projection from embedding space. Projection-based utilities should be treated as experimental unless calibrated.

**Status:** Mean, persona and contrastive high-level paths use unsteered model activations. The embedding-projection extractors were removed in improvement-plan Phase 3 (M6). Centering/normalization and model-specific output checks still matter: using model-native activations alone does not establish semantic selectivity.

### 4. Add a workspace audit mode

A future CLI mode should compare baseline and steered generations across prompts and report religious-reference intensity, refusal changes, length shifts, and diagnostics.

**Status:** Implemented as `experiments/steering_eval.py` ([evaluation guide](evaluation.md)). It compares a baseline with steered and retrieval conditions on held-out English and Arabic prompts and reports script shifts, degeneration, NLL under the unsteered model, ARC-Easy accuracy, an embedding thematic proxy, transport and dose attainment, with bootstrap intervals and provenance. Religious-reference intensity is judged by human raters on a blinded sheet rather than by keyword counts; refusal changes are not yet measured.

### 5. Add counterfactual reflection experiments

Counterfactual reflection should be implemented in `experiments/`, not as default runtime behavior. The experiment should compare normal prompts with interrupted reflection prompts and build contrastive vectors from the differences.

**Status:** Planned.

### 6. Add a structured concept vocabulary

Move Quranic bridge themes into structured concept objects containing canonical names, aliases, Arabic/Islamic terms, related themes, and safety notes.

**Status:** Planned.

### 7. Add oversteering safeguards

Future work should cap relative perturbation size per layer and surface warnings when coefficients cause broad hidden-state shifts.

**Status:** Partially implemented. Runtime validation now rejects nonfinite vectors/configuration, blend coefficients outside [0, 1], and invalid dimensions. Hook state is serialized and restored, and clamp receives its intended coefficient. High-level add-mode doses are now target relative perturbations per layer, calibrated from median token norms, and diagnostics report the achieved ratio. Automatic caps and calibrated behavioral warning thresholds remain planned. A small perturbation is not a universal safety guarantee; the centered SmolLM2 results still show generation collapse at a target ratio of at most 0.1. Such research thresholds cannot override guardian policy.

### 8. Clarify documentation

The README should distinguish retrieval grounding, activation intervention, CAA, and workspace-style interpretability.

**Status:** Implemented through the README, architecture, steering and testing guides. The research note separates measured model evidence from synthetic gateway tests, and the containment plan identifies the live-host work still pending.

