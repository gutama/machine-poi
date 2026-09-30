# Geometry literature and containment research

Reviewed 2026-09-30 against primary-source metadata and abstracts. This replaces
the earlier analogy-led prioritization; it does not reproduce the papers' results.

[Research design](containment_geometry_research.md) · [Research directions](research_directions.md) ·
[Workspace roadmap](global_workspace_improvement_plan.md)

## What the sources support

| Primary source | Established scope of the source | Use in Machine-POI | Transfer limit |
| --- | --- | --- | --- |
| You, Deng & Chen, [Spherical Steering](https://arxiv.org/abs/2602.08169), 2026 | Activation rotation preserving hidden-state magnitude | Essential comparator for the proposed low-rank `Cl(r,0)` rotor | Rotation/norm preservation is existing work; tool containment is not established |
| Oozeer et al., [Riemannian-Manifold Steering](https://arxiv.org/abs/2605.24942), 2026 | Learned approximation to a behavioral pullback metric; arithmetic steering experiments | Later learned-metric comparator after spherical/additive baselines | Arithmetic class control does not validate unsafe-action prediction |
| Cho et al., [Curve Your Attention: Mixed-Curvature Transformers for Graph Representation Learning](https://arxiv.org/abs/2309.04082), 2023 | Product constant-curvature spaces for graph Transformers | Precedent for the deferred product-space architecture experiment | Graph reconstruction/classification results are not agent or pretrained-LLM evidence |
| Ji, [RiemannFormer](https://arxiv.org/abs/2506.07405), revised 2025 | Attention involving metrics, tangent spaces and parallel transport | Compare explicit metric/frame contracts when changing attention | A post-hoc query/value wedge is not automatically the paper's transport operator |
| Di Sipio, Diaz-Rodriguez & Serrano, [The Curved Spacetime of Transformer Architectures](https://arxiv.org/abs/2511.03060), 2025 | Geometric analogy and representation-trajectory turning/deflection experiments | Cheap trajectory-shape controls under context edits | Extrinsic turning, intrinsic curvature and agent safety are different claims |
| Arditi et al., [Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), 2024 | Causal interventions on a refusal direction in studied chat models | Motivate refusal and harmless-task controls alongside compliance | Refusal is not authorization; refusing everything can destroy utility |
| Debenedetti et al., [AgentDojo](https://arxiv.org/abs/2406.13352), 2024 | Tool-agent tasks, injected tool responses and extensible attacks | Primary trajectory benchmark | Policy adapters and tool-capable models must be validated separately |
| Zhan et al., [InjecAgent](https://arxiv.org/abs/2403.02691), 2024 | Indirect injection benchmark for tool-integrated agents | Complementary attack/task families | Benchmark success is not coverage of host bypasses or remote cancellation |
| Debenedetti et al., [CaMeL: Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813), 2025 | Separates trusted control/data flow and enforces capability policies | Strong security architecture comparator | Machine-POI's scope gateway does not implement the same data-flow semantics |
| Ames et al., [Control Barrier Function Based Quadratic Programs](https://arxiv.org/abs/1609.06408), 2017 | Forward invariance under specified continuous dynamics and controller conditions | Optional toy controller theory after declaring dynamics/safe set | Those conditions are not supplied by LLM embeddings or empirical safety probes |

## Mathematical corrections before further literature transfer

The current diagnostics construct skew generators, adjacent differences,
commutators and products of three vertex rotations. They do not supply a declared
curvature two-form or closed loop of directed edge transports. A constant
commuting generator can produce a nonzero legacy “holonomy” angle. Query and
value coordinate spaces also admit independent reparameterizations.

The [research design](containment_geometry_research.md#4-repair-the-meaning-of-transport-diagnostics-first)
specifies common frames, inverse links, loop orientation and pure-gauge controls.
Run `python experiments/geometry_sanity.py` for an independent GA/vector check and
the constant-generator counterexample. The executed checks validate those small
fixtures only. Historical runtime fields and reports remain unchanged.

Do not equate these quantities without a derivation:

- a representation path's turning angle or length-to-chord ratio;
- the statistical/Riemannian curvature of a declared metric;
- noncommutativity of selected skew generators;
- closed-loop transport holonomy;
- independently scored model behavior and committed tool effects.

A curve can bend in flat Euclidean space. A frame field can vary while its links
remain pure gauge. Low curvature can accompany unsafe actions. Each inference
requires its own measurement and controls.

## Recommended reading and implementation order

1. Spherical Steering plus existing additive baselines: establish a fair,
   bounded-intervention comparison before inventing a new curved architecture.
2. AgentDojo, InjecAgent and CaMeL: build a receipt-scored agent evaluation and
   identify where structured scope checks miss harmful permitted data flows.
3. Riemannian-Manifold Steering: investigate a local SPD/pullback metric only
   if simple rotor/additive comparisons justify the cost.
4. RiemannFormer and mixed-curvature graph Transformers: use later to design
   actual metric-compatible attention or product-space components.
5. Continuous barrier-function theory: keep its assumptions explicit in a toy
   system; do not cite it as a proof of discrete agent containment.

Older roadmap entries and harvested literature rankings are research leads.
Their metadata, benchmark scope and claimed equivalences need source-level
verification before use in a publication. Topic growth or a general-relativity
analogy is not evidence for a safety mechanism.
