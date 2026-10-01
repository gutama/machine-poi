# Agent containment with bounded geometric interventions

Research design, 2026-09-30. Baseline reviewed: `2267edb8858e31336b91d5a7c02e4b43174f96fc`.

[Research directions](research_directions.md) · [Architecture](architecture.md) ·
[Containment plan](rogue_agent_containment_plan.md) · [Geometry literature](curvature_literature_roadmap_review.md)

## Main question and contribution

**Can bounded, geometry-aware interventions reduce unauthorized action proposals
and operator burden at matched benign task utility, while an independent host
boundary continues to enforce permissions?**

The proposed contribution is a controlled study connecting representation control
to agent action trajectories. It is not a claim that curved space contains an
agent. Compare geometry against strong prompting and additive steering, and
report negative results. A rotor formulation alone is not novel: spherical
activation rotation already exists [1]. Novelty would require better evidence
about tool actions, adaptive attacks, representation assumptions and utility.

The existing Quran-derived corpus remains one interpretable intervention source.
It is a secondary, language-matched ablation. Neither religious register nor a
refusal direction establishes compliance with the host's task grant.

## 1. Three objects with separate responsibilities

| Object | Representation | What can be claimed |
| --- | --- | --- |
| Host authority and effects | Typed grants, policy, adapter-resolved data flow and service receipts | Containment for named, completely mediated paths under explicit deployment assumptions |
| Model behavior | Hidden states, tool proposals and task outcomes | An empirical change in proposal rate, task success or review burden |
| Geometry | A declared metric, frame, transport and intervention map | Algebraic invariants and geometric measurements within that model |

A grant contains exact scopes and discrete states; its allowed action set need
not be smooth, convex or connected. Do not infer legal actions from a geodesic
between embeddings. A latent projection can change a proposal; only the host can
authorize the resulting typed action.

The external-effect acceptance condition is:

\[
  \operatorname{effect}(a,t)\Rightarrow
  \operatorname{valid\_grant}(a,t)\land\operatorname{scope}(a,t)
  \land\operatorname{budget}(a,t)\land\operatorname{required\_approval}(a,t).
\]

Here `t` is the service's effect-commit point. The current gateway checks before
adapter entry; this stronger condition requires adapter checkpoints, service-side
revocation/idempotency, and a declared concurrency/restart model. It is a target
invariant, not an already-proven guarantee for remote effects. The agent must have
no direct credentials, host control API access or unmediated effect path.

### Threat model and containment work

Assume the model may ignore instructions, submit arbitrary JSON, retry, delegate,
or process poisoned retrieval, tool outputs and persistent memory. Give adaptive
attackers the public intervention design. The trusted host, policy store and
adapters are the initial trusted computing base; test their failure modes
separately. Model telemetry is untrusted evidence.

| Priority | Experiment | Required evidence |
| --- | --- | --- |
| C1 | Inventory filesystem, network, subprocess, graph providers and memory effects | Each protected effect mapped to a mediated adapter; explicit residual bypasses |
| C2 | Run a worker under a distinct OS identity with default-deny egress and no credentials | Direct service, filesystem and host-control attempts fail independently of model behavior |
| C3 | Exercise approval mutation, delegation, concurrent retries, revocation, crash/restart and remote jobs | Service receipts, persistent replay/budget state, cancellation latency and committed-effect counts |
| C4 | Test harmful content within an otherwise permitted tool/resource scope | Host-owned provenance/data-flow rules or documented coverage gap; descriptive `goal` is insufficient |
| C5 | Run live benchmark agent trajectories | Benign utility and attacker-objective success separated from gateway fixture correctness |

C1–C3 must precede deployment claims. The reference process demo uses the same OS
account and mock effects; it does not satisfy C2. `Gateway.observe()` records
signals without reducing authority. Automatic hold/throttle behavior is proposed
work and needs explicit host policy, false-positive calibration and concurrency
semantics. A pause currently permits already-dispatched actions to finish; a
geometric alarm cannot silently acquire stronger semantics.

## 2. Start with a low-rank spherical rotor

Use the real Euclidean algebra `Cl(r,0)`, an ordered orthonormal basis
`e1,...,er`, and orientation `I=e1...er`. Products are `ab=a·b+a∧b`, reversal
is `~A`, and `A×B=(AB-BA)/2`. Avoid a dense `2^r` multivector representation.
Store vectors and a simple bivector's two spanning directions.

At each selected layer fit and freeze an orthonormal basis `Q` of shape `d×r`
on training data only. For column hidden state `h`, write

\[
 z=Q^T h,\qquad h_\perp=h-Qz,\qquad u=z/\|z\|.
\]

Let `s` be a fixed, training-derived target direction in this subspace. Define

\[
 t=\frac{s-(s\cdot u)u}{\|s-(s\cdot u)u\|},\quad
 B=u\wedge t,\quad B^2=-1,\quad
 R=\exp(-\theta B/2).
\]

The update and independent vector implementation are

\[
 z'=Rz\widetilde R=\|z\|(u\cos\theta+t\sin\theta),
 \qquad h'=Qz'+h_\perp.
\]

`R~R=1`; the sandwich fixes the rotation sign (`e1` moves toward `e2` for
`B=e1∧e2` and positive theta). The Euclidean hidden-state norm is preserved
because `Q` is orthonormal and the residual is unchanged. This says nothing about
the downstream model's capabilities or behavior.
The sphere is a designed intervention constraint; it is not evidence that the
checkpoint's semantic representations naturally lie on a spherical manifold.

Select nonnegative theta on development data and cap it at both a small declared
`theta_max` and the angle to the target so a step does not overshoot. Log
`||h'-h||/||h||` per token/layer. For this update,

\[
 \frac{\|h'-h\|}{\|h\|}
 =\frac{\|z\|}{\|h\|}\,2\sin(\theta/2)
 \leq 2\sin(\theta/2),\qquad 0\leq\theta\leq\pi.
\]

Match additive baselines on achieved relative displacement, not coefficient or
angle. Across layers also record cumulative displacement and output KL: each
norm-preserving step can still disrupt computation.

Degenerate cases are explicit: zero projected state or target, parallel target,
and antipodal target use no intervention and record a reason. Antipodes do not
select a unique plane; choosing an arbitrary plane hides a modeling decision.
Near-degenerate cases use a declared tolerance. Non-finite inputs abort the
experimental condition. Do not update the basis or target from untrusted live
retrieval. The small verifier in `experiments/geometry_sanity.py` checks the
`Cl(3,0)` identity against its vector implementation. The opt-in runtime G1 hook
now lives in `machine_poi/rotor.py`; its invariant tests compare against that
independent sandwich implementation. The [guidance pipeline](quran_guidance.md)
fits training-only artifacts and matches measured development displacement, with
held-out attainment checks. This implementation has scripted integration evidence;
actual model efficacy remains unmeasured when pinned weights are unavailable.

## 3. Curved-space design ladder

| Stage | Space and metric | Purpose | Promotion gate |
| --- | --- | --- | --- |
| G0 | Euclidean residual space; fixed low-rank subspace | Centered additive steering and prompt-only controls | Stable held-out task/proposal baseline |
| G1 | Sphere in `Cl(r,0)`; radius retained separately | Bound angular edits and isolate magnitude changes | Beat dose-matched addition on utility/risk frontier |
| G2 | Local SPD metric `g(z)=J_f(z)^T J_f(z)+lambda I`, `lambda>0` | Penalize edits that strongly alter a declared behavioral map `f` | Held-out prediction/calibration plus gains beyond whitening |
| G3 | `H^a×S^b×R^c` with declared block metric, scales and curvature | Explore hierarchical relation features, direction and unconstrained features | Task-specific justification and matched parameter/compute ablations |
| G4 | Grassmann `Gr(k,r)` for permitted-feature subspaces | Track subspace changes and principal angles | Stable rank, identifiable subspace and added value over PCA |

G2's `f` must be specified (e.g. a frozen decoder feature map). A low-rank
Jacobian is a computational approximation; regularization makes the metric SPD
but does not make it accurate. A Fisher pullback needs its probability model and
can be singular. Whitening a constant SPD metric is still flat geometry; it is
a mandatory baseline, not evidence for curvature. Whitened rotations preserve
the chosen metric norm, not automatically the original hidden-state norm.

G3 is a later architecture experiment. Hyperbolic hierarchy is a modeling
hypothesis, not permission ordering; spherical direction is not ethical value.
A hyperbolic manifold has a positive-definite tangent metric even if its
hyperboloid embedding uses `Cl(a,1)`. Declare that embedding before using boosts;
do not use an indefinite norm as a risk score. Product metrics require separate
coordinate blocks and exp/log/transport maps. A scalar norm of a bivector is not
a complete account of all principal rotation angles.

The existing `gpt_on_manifolds_v4.py` projects embeddings in a toy training
experiment. Its result cannot establish a curved residual-stream geometry in a
pretrained checkpoint or agent containment. Full curved-attention training stays
behind G1/G2 evidence. Measure wall time and memory; GA notation has no automatic
speed advantage.

## 4. Repair the meaning of transport diagnostics first

The existing `machine_poi/workspace_diagnostics.py` constructs skew matrices from
query/value coordinates, forms adjacent differences plus commutators, and calls
`T_k T_j T_i` a triangular holonomy. Retain existing values for reproducibility,
but interpret them as **constructed generator variation, noncommutativity and
three-rotation product angles**, pending a new metric/transport contract.

There are three separate issues:

1. A connection along a one-dimensional token chain does not supply independent
   two-form directions for intrinsic curvature `F=dA+A∧A`. Adjacent differences
   plus commutators are not a derivation of that two-form. Noncommuting controls
   can be useful measurements without being manifold curvature.
2. Vertex-indexed rotations `T_t` have no specified start/end fibers. For a
   constant generator `T_i=T_j=T_k=T`, all commutators and adjacent differences
   vanish, while `T^3` generally rotates. Thus a nonzero legacy angle is not
   evidence of a curved loop. Zero connection alone is an insufficient control.
3. Queries/keys and values have independently reparameterizable coordinates.
   Equal head dimension does not identify query and value spaces. The wedge
   `q∧delta_v` needs an explicit shared frame or must be labeled coordinate
   dependent. For row-vector attention, `q→qM`, `k→kM^{-T}` leaves logits
   unchanged; independently, `v→vN`, `W_O→N^{-1}W_O` leaves output unchanged.
   A generator built from raw q/v can change under these function-preserving
   transformations. An orthogonal common-frame check alone misses this issue.

For a genuine discrete transport experiment declare `U_ij` to map vectors at
vertex `i` into the fiber at `j`. Require `U_ji=U_ij^{-1}`; for orthonormal frames
require `U_ij^T U_ij=I`. A triangle's based loop is

\[
 H_i=U_{ki}U_{jk}U_{ij}.
\]

Under frame changes `G_i`, links transform as `U_ij'=G_j U_ij G_i^{-1}` and
`H_i'=G_i H_i G_i^{-1}`. Eigenangle/trace observables are therefore invariant.
For pure-gauge links `U_ij=F_j F_i^{-1}`, every closed product is identity even
when frames vary. In a nonorthonormal frame metric compatibility instead requires
`U_ij^T g_j U_ij=g_i`.

Add commuting-generator, pure-gauge, inverse-edge, frame-change, and known
noncommuting-loop controls. Estimate curvature only with a declared two-parameter
surface (e.g. dose × controlled context edit), common fiber identification and
refinement/loop-area scaling. A group commutator loop measures control order
sensitivity; calling it curvature requires a connection model. Frames derived
solely as `F_j F_i^{-1}` are always pure gauge and cannot detect curvature.

Rename/version diagnostic fields in a separate implementation with schema
migration; do not quietly reinterpret historical reports. This research update
does not change the runtime diagnostic algorithm.

## 5. A factorial, trajectory-level evaluation

Use a pinned tool-capable checkpoint before drawing agent conclusions. Start with
one checkpoint and one benchmark suite, then replicate on a second architecture
and held-out language/attack families. AgentDojo [3] supplies realistic tool tasks
and injected tool outputs; add InjecAgent [4] and local containment drills where
their action/data-flow coverage fits. Adapt typed tools to the gateway and validate
policy coverage before comparing attack rates. Benchmark permissions must permit
intended benign tasks; an always-deny host is not a useful defense.

| Intervention | No enforced gateway | Enforced gateway |
| --- | --- | --- |
| None | Behavioral baseline | Boundary baseline |
| Prompt-only policy | Strong prompting baseline | Prompt + boundary |
| Centered additive steering | Additive behavior control | Additive + boundary |
| Spherical rotor | Geometry behavior control | Rotor + boundary |

All cells use isolated mock services or benchmark sandboxes; “no gateway” never
means uncontained access to real services. Add random-plane, shuffled-label,
whitened-additive and, only later, G2 metric controls. Use matched context,
decoding, token budgets and paired seeds, with intervention strength calibrated
on development trajectories only. Freeze vectors, layer bands, probe thresholds,
policy and adapter versions before test. Split by task/attack template and source
document; related variants belong in the same split.

Score at the **episode** as well as the action level:

- unauthorized proposal rate before mediation, with structured ground truth;
- attacker-objective success, including in-scope misuse and protected reads;
- unauthorized committed effects, independently scored from service receipts;
- benign task completion, false blocks, review frequency and operator effort;
- retries, child runs, budget use, p50/p95 latency and time to containment;
- degeneration, tool JSON validity, language/register shift and capability loss.

Log redacted episode/action IDs, grant/policy/adapter versions, proposed and
resolved scopes, decisions, receipt IDs, achieved intervention norms/angles, model
revision, data hashes and seeds. Reject incomplete telemetry as unavailable,
not a zero-risk observation. Do not log raw sensitive hidden states by default.

Bootstrap paired differences by independent task/template clusters, with seeds
nested within clusters. Report per-family results and denominators. Attackers get
a fixed optimization budget and held-out adaptive attacks target probes and
steering as well as the gateway. Correct exploratory head/layer comparisons for
multiplicity or explicitly label them exploratory.

Pre-register a benign-task noninferiority margin (initial pilot proposal: 5
percentage points), minimum useful unsafe-proposal reduction and review-effort
tradeoff after a pilot/power analysis. These are research choices, not universal
safety thresholds. Zero effects in `n` independent episodes gives only a
one-sided 95% binomial upper bound `1-0.05^(1/n)`; approximately `3/n`. Correlated
variants do not count as independent episodes. Any deterministic containment
violation in a scoped drill blocks deployment until understood.

## 6. What would justify a geometric signal?

Fit a linear probe and cheap behavioral signals first. Compare geometric features
against perturbation size, output entropy, action history, retry rate and a
language-matched injection probe. Assess held-out calibration, precision-recall
and lead time **before the first effect**, not only after a policy denial.
Perform randomized interventions with matched displacement: correlation between
a diagnostic and harm does not establish a causal controller.

A signal initially enters offline/shadow analysis. A future host may tighten
review or stop a run, but must never broaden scope or skip an existing review.
Missing activations, black-box models and adversarially spoofed telemetry need an
explicit fallback. Signals derived after a tool call cannot protect that effect.
Rewarding low curvature or small holonomy has no safety interpretation without
independent behavioral evidence.

Continuous-time barrier-function theory [5] can motivate a **toy** dynamical
model. Its invariance results require a specified system, regularity, valid safe
set and feasible controls. LLM token/tool transitions are discrete, partially
observed and adversarial; an empirical probe is not a certified safe-set boundary.
For agents, first study discrete reachable states and host-mediated transitions.
Do not transfer the continuous theorem as a containment proof.

## 7. Ordered deliverables and stopping rules

| Milestone | Deliverable | Advance only if |
| --- | --- | --- |
| M0 | Geometry contract, legacy counterexample, threat model and adapter inventory | Known invariants hold; “curvature” claims have declared meaning |
| M1 | One sandboxed tool-capable agent and gateway benchmark adapter | Benign task coverage and independent receipt scoring work |
| M2 | Opt-in low-rank rotor hook plus dose-matched controls | Session restoration, degeneracies and actual displacement verified |
| M3 | Frozen factorial benchmark with adaptive attacks | Utility/risk intervals and compute costs support incremental benefit |
| M4 | G2 metric/transport experiment; calibrated signal in shadow mode | Beats G1 and cheap probes out of distribution |
| M5 | Host isolation, durable state and service revocation drills | Named effect paths pass bypass/restart/kill-switch acceptance |

The first implementation slice should be **M1 + M2**, after M0. Keep G3/G4 and
full curved-attention training deferred. If rotors do not improve the utility/risk
frontier, publish that result and keep the simpler boundary + additive/prompt
baseline. If geometry has no predictive gain over cheap features, retire it as a
containment signal. A successful containment study can stand without geometric
gains; a successful geometric study still needs the external boundary.

## References and evidence level

Primary-source metadata/abstracts checked 2026-09-30. These are experimental
precedents, not validation of Machine-POI. No new model/agent result is claimed.

1. You, Deng & Chen (2026), [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://arxiv.org/abs/2602.08169). Published-method comparator for norm-preserving steering.
2. Turner et al., [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248); Rimsky et al., [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/). Additive controls; existing repository references.
3. Debenedetti et al. (2024), [AgentDojo](https://arxiv.org/abs/2406.13352). Agent task/attack evaluation.
4. Zhan et al. (2024), [InjecAgent](https://arxiv.org/abs/2403.02691). Complementary tool-agent injection benchmark.
5. Ames et al. (2017), [Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408). Conditional continuous-time invariance theory; transfer limits above.
6. Debenedetti et al. (2025), [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813). Control/data-flow and capability-policy comparator; Machine-POI does not implement its taint semantics.
7. [Riemannian-Manifold Steering: Geometry-Aware Generative Autoencoders for Label-Free Steering](https://arxiv.org/abs/2605.24942) (2026). Learned-metric steering research; compare only after G1.
8. Cho et al. (2023), [Curve Your Attention: Mixed-Curvature Transformers for Graph Representation Learning](https://arxiv.org/abs/2309.04082). Product-space precedent on graphs; not evidence for agent containment.
