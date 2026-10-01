# Machine-POI: Activation Steering and a Host-Side Action Boundary

Research and implementation note, updated 2026-09-29.

[Architecture](docs/architecture.md) · [Steering guide](docs/steering_guide.md) ·
[Validation](docs/testing.md) · [Committed model results](experiments/results/README.md)

## Abstract

Machine-POI is a Python research implementation of Quran-derived activation
steering and hierarchical retrieval. It constructs mean-activation, weighted
persona and contrastive directions, injects them through model hooks, and reports
activation and attention-transport diagnostics. A separate standard-library
reference gateway evaluates structured agent tool proposals against host-issued
grants, with bound operator review, budgets, replay prevention and revocation.

These components address different questions: steering experiments examine
changes in generated behavior; the gateway constrains authorized effects at a
trusted host boundary. On a held-out evaluation of a 0.5B model, stronger steering
traded capability for religious register and degenerated at the highest dose, so
this note does not claim preserved general capabilities or reliable moral
alignment. Gateway validation consists of mocked
regression tests and synthetic proposed-action fixtures. Live-host protection and
held-out agent/model comparisons remain unmeasured.

## 1. Scope

Changing a model's style or thematic emphasis does not establish authority to use
a tool. Machine-POI therefore keeps behavioral interventions separate from the
host's task grant. Model output, retrieved instructions and diagnostic scores
cannot increase that grant.

The research implementation supports controlled comparisons without changing
model weights. Its verse/passage/surah organization is text chunking, termed
multi-resolution analysis (MRA) in the code. It does not imply a wavelet transform
or a proven hierarchy of semantic concepts.

## 2. Methodological context

### 2.1 Activation engineering

Activation Addition modifies intermediate model activations using directions
constructed from contrasting prompts [1]. Machine-POI uses forward hooks and
several vector recipes. Its default corpus-mean path is centered on a neutral
control set, a contrast in the spirit of CAA [2]; the older uncentered mean
(`recipe="raw_mean"`) is not the same protocol as the original
contrasting-prompt experiments.

### 2.2 Contrastive activation addition

CAA constructs steering directions from positive/negative activation differences
[2]. `ContrastiveQuranSteerer` pools examples at each layer, subtracts the two
means, and normalizes the difference. The default negative set is neutral Modern
Standard Arabic prose written for the project, so the contrast is not also
Arabic-versus-English. It can still mix register, topic and behavior. Attributing
an effect to a particular value requires controls that separate those factors.

### 2.3 Retrieval

Retrieval-augmented generation supplies external context [3]. Machine-POI uses
ChromaDB and optional LightRAG to retrieve text and graph context. This changes
the model input. Dynamic steering additionally constructs a temporary activation
intervention from retrieval; that path is off by default and requires explicit
trusted-corpus opt-in.


## 3. Related work

### 3.1 Representation steering and representation geometry

[Representation Engineering (RepE)](https://arxiv.org/abs/2310.01405) studies model behavior through population-level representations and shows that high-level concepts and behaviors can be monitored and manipulated through activation-space interventions. Machine-POI follows this general representation-steering paradigm, but applies it to a Quran-derived corpus with neutral Arabic controls and evaluates the resulting intervention together with capability, likelihood, degeneration and thematic measurements rather than treating representation manipulation itself as a safety guarantee.

[The Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658) provides a geometric account of how concepts can be represented approximately linearly in neural representations and develops a causal framework for reasoning about interventions in representation space. Machine-POI uses activation directions empirically rather than testing the hypothesis itself: its centered Quran-derived vectors are intervention tools, and the experiments measure their behavioral and capability effects.

[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) shows that refusal behavior in open-source language models can be strongly mediated by a single activation direction: adding the direction can induce refusal, while removing it can suppress refusal. Machine-POI differs in both target and purpose: its steering direction is derived from Quranic material rather than refusal behavior, and its experiments do not claim to control or improve refusal behavior.

[AxBench](https://arxiv.org/abs/2501.17148) evaluates a broad range of representation-based concept detection and steering methods against prompting and fine-tuning across multiple models and concepts. Machine-POI is narrower: it is not intended as a general steering benchmark, but studies centered Quran-derived directions, Arabic controls, dose calibration and capability trade-offs in a specific 0.5B model setting.

[Persona Dosing](https://arxiv.org/abs/2609.36388) studies how activation steering can be calibrated to produce specified intensities of persona-related behavior, separating the behavioral range achievable by a controller from the accuracy of requested intensities. Machine-POI also calibrates intervention strength, but its dose is defined as a relative activation perturbation rather than a validated human-interpretable behavioral intensity; its goal is to measure the effects of a Quran-derived direction rather than to provide controllable persona expression.

[A Geometric Account of Activation Steering through Angle-Norm Decomposition](https://arxiv.org/abs/2606.06735) argues that additive activation interventions simultaneously change angular alignment and hidden-state norm, and studies these effects separately across several steering methods and models. Machine-POI currently reports activation norms, cosine alignment and relative perturbation, but does not decompose its steering effects into the separate angular and radial components proposed by this account.

### 3.2 Language and cultural alignment

[Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) studies whether multilingual models rely on English-shaped internal representations even when processing other languages, finding evidence that key decisions can remain close to English in representation space and that English-derived steering vectors can be more effective. Machine-POI instead studies an Arabic-specific intervention and explicitly uses neutral Arabic controls to reduce the possibility that the measured effect is simply an Arabic-versus-English difference.

[LangFIR](https://arxiv.org/abs/2604.03532) identifies sparse language-specific features from monolingual data using sparse autoencoders and uses them for language steering. Machine-POI does not attempt to identify language-specific SAE features; its centered Quranic direction is intended to capture differences between Quran-derived material and neutral Arabic prose, so language identity is treated as a control rather than the target of steering.

[Steering Multilingual Models Towards Cultural Knowledge](https://arxiv.org/abs/2605.23069) investigates inference-time activation steering for multilingual cultural-knowledge tasks using language vectors extracted from parallel data, and reports that steering effects vary across layers, language-region pairs and prompt formulations. Machine-POI studies a different intervention: rather than steering a model toward general cultural knowledge, it derives directions from Quranic material and evaluates whether those directions change religious register and related behavioral measures while tracking capability costs.

[Investigating Cultural Alignment of Large Language Models](https://arxiv.org/abs/2402.13231) studies cultural alignment through model responses to culturally grounded questions and compares those responses with survey data from human participants, including experiments involving Arabic and English. Machine-POI differs by intervening directly in the model's activation space rather than primarily measuring cultural alignment through simulated surveys or changing the model's training-data mixture. Its neutral Arabic control set is intended to help distinguish Quran-related effects from general Arabic-language effects, although topic, register and cultural confounds remain possible.

### 3.3 Agent containment

[Out-of-Band Policy Enforcement at a Trusted Tool Boundary](https://arxiv.org/abs/2608.27646) is particularly close to Machine-POI's host-side component. It places a trusted enforcement boundary outside agent reasoning so that authorization and policy decisions are made independently of the model, including restrictions on operations, resources and returned information. Machine-POI follows the same central separation between model behavior and authority: the authenticated host creates exact-scope grants and the guardian rechecks authorization before execution. It is substantially narrower, however: the current Machine-POI implementation is a standard-library reference gateway with in-memory state and synthetic action fixtures, rather than a general trusted proxy with the broader policy and information-flow enforcement evaluated by OBPE.

[Progent](https://arxiv.org/abs/2504.11703) addresses least-privilege control for tool-using agents by representing privileges as symbolic policies over tools and arguments and deterministically checking proposed tool calls against those policies. It also uses solver-based policy updates to ensure that an agent's effective action space cannot expand without approval. Machine-POI similarly prevents model output from increasing a host-issued grant, but uses fixed grants, scope and state checks rather than Progent's symbolic policy representation and solver-based privilege-update mechanism.

[CaMeL, in *Defeating Prompt Injections by Design*](https://arxiv.org/abs/2503.18813), separates trusted control information from untrusted data and uses capability-based mechanisms to prevent prompt-injected content from directly controlling privileged operations or causing unauthorized data flows. Machine-POI shares the principle that model-generated text should not itself determine authority, but does not implement CaMeL's control/data-flow analysis or capability propagation and therefore does not claim equivalent prompt-injection protection.

[Cordon](https://arxiv.org/abs/2606.17573) introduces a transaction-oriented approach to agent safety in which multi-step tool use is represented as a semantic transaction with effect lineage, validation and staged external effects before commitment. Machine-POI instead enforces authorization at the structured-action boundary, with expiry, state, budget, review, replay and cancellation checks; it does not currently provide Cordon's transaction-wide lineage or commit/rollback model.

[ContainmentBench](https://arxiv.org/abs/2607.23999) focuses on evaluating agent containment after an agent has been exposed to malicious content. Its trace-based evaluation considers propagation, recovery and subsequent actions rather than only whether the final endpoint was compromised. Machine-POI currently has synthetic gateway fixtures for benign and forbidden proposed actions, but those fixtures begin with structured proposals and do not run an LLM or model an actual prompt-injection trajectory, so they do not establish the type of post-exposure containment measured by ContainmentBench.

[AgentDojo](https://arxiv.org/abs/2406.13352) provides a dynamic benchmark for tool-using agents exposed to indirect prompt injections in stateful environments. It evaluates realistic multi-step tasks in which malicious instructions can arrive through tool outputs and subsequently influence tool use. Machine-POI's current gateway tests are deliberately smaller and begin after an action proposal has already been produced; AgentDojo therefore represents a natural future environment for testing whether the host-side boundary remains effective when proposals originate from an agent operating under prompt injection.

[InjecAgent](https://arxiv.org/abs/2403.02691) provides a benchmark specifically targeting indirect prompt injection against tool-integrated LLM agents, with attack scenarios involving malicious external content and unauthorized agent actions. Machine-POI does not currently reproduce these attack scenarios: its containment evidence consists of synthetic proposed-action fixtures rather than live LLM agents exposed to injected tool content.

Together, these lines of work motivate the separation used by Machine-POI. Activation steering changes model behavior and can introduce capability or stability trade-offs, while the guardian is intended to constrain external effects independently of those behavioral changes. The current implementation should therefore be understood as a combination of a Quran/Arabic activation-steering study and a reference host-side authorization boundary, rather than as a general prompt-injection defense or evidence that steering itself provides agent containment.


## 4. Implemented steering methods

### 4.1 Vector construction

For layer `l`, let `h_l(T_i, t)` denote an unsteered decoder-layer output at token
`t` of sample `T_i`. The mean-activation recipe first pools within each text,
then averages across sampled texts:

$$
\mu_l = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{|T_i|}
\sum_{t=1}^{|T_i|} h_l(T_i,t), \qquad v_l = \operatorname{normalize}(\mu_l - \nu_l),
$$

where `ν_l` is the same pooled mean over the 120 neutral Arabic control sentences.
Without centering, `μ_l` is dominated by the component that every hidden state
shares, so its direction says little about the corpus; `recipe="raw_mean"` keeps
`v_l = normalize(μ_l)` for reproducing older results and warns.

Each text receives equal weight after token pooling. The token sum runs over
content tokens: BOS and other special tokens are excluded by default, since the
first position carries a large generic activation shared by every text. The high-level persona path
computes a separate centered direction for verses, paragraph chunks of up to 19 verses
within a surah, and surahs, combines them with default weights 0.50/0.35/0.15, and normalizes the
combined vector. Paragraphs and surahs are much longer than the control sentences,
so their contrast also carries length. The contrastive path normalizes `mean(positive) - mean(negative)`.
Zero norms are handled by the underlying normalization routines; a zero vector
has no semantic direction.

The raw-vector experiments in `experiments/` can use different normalization and
dose conventions. Their coefficients must not be substituted directly into the
high-level normalized persona API.

### 4.2 Intervention

Hooks act on decoder-layer **outputs**, not directly on the
`post_attention_layernorm` submodule. The hook supports tensor or tuple outputs
and applies its intervention at all positions present in each forward call.
Let `a` be its coefficient, `v` its supplied vector, and
`u = v / (norm(v) + 1e-8)`:

| Mode | Hidden-state update |
| --- | --- |
| Add | `h' = h + a*v` |
| Blend | `h' = (1-a)*h + a*v` |
| Replace | `h' = v` |
| Clamp | `h' = h - dot(h,u)*u + a*u` |

The high-level API scales the coefficient by layer distribution; for replacement
it scales the vector before registration. Clamp controls a projection rather
than an additive dose. Zero clamp removes that projection and is not a baseline.
No mode has been shown here to preserve fluency at arbitrary strength.

In add mode the high-level dose is a target relative perturbation `r`: layer
`l` uses `a_l = r * scale_l * n_l / norm(v_l)`, where `n_l` is the median
per-token hidden-state norm on neutral calibration sentences. A raw coefficient
does not transfer between models, whose activation scales differ by more than an
order of magnitude; a ratio does. The median avoids the first-position
attention-sink token, whose norm dominates a mean.

### 4.3 Retrieval and domain bridges

MRA adds verse, passage and surah context to the prompt. Domain bridging first
tries static keyword/theme mappings, then optional graph traversal, then an
embedding-similarity fallback. Graph generation combines graph answers, selected
verses and bridge terms. Retrieved content is bounded and quoted as reference
data. An explicit trust flag permits dynamic retrieval steering for experiments;
it does not verify the corpus or detect malicious instructions.

## 5. Runtime and diagnostics

`SteeredLLM` serializes inference and hook mutations. High-level generation scopes
temporary steering to a session, restores prior vectors/modes/enabled flags on
success or failure. Registration replaces a layer's previous handle. Activation
pooling runs in right-padded batches with steering disabled and removes its
hooks in `finally`. Async graph retrieval finishes before entering
the synchronous session.

Pointwise diagnostics report activation/vector norms, cosine alignment,
projection magnitude and relative perturbation computed from the actual update
for each injection mode, averaged over every steered token of a generation. The
achieved dose ratio divides the mean update norm by the median token norm.
High-level generation retains these scalar summaries in `last_run_diagnostics`. Attention-transport experiments summarize a constructed
connection using variation/commutator terms and holonomy. Those quantities are
research diagnostics, with no validated threshold for authorization or safety.

Remote model code defaults off and requires a pinned commit for explicit opt-in.
Steering caches store numeric arrays and identity metadata rather than pickled
objects. Metadata detects accidental reuse, not malicious artifact forgery.

## 6. Experimental evidence

Steering claims in this note cite one run of the evaluation harness,
`experiments/steering_eval.py` ([evaluation guide](docs/evaluation.md); results in
[`experiments/results/`](experiments/results/README.md)). The earlier probes in that
folder predate the harness. They used 1-16 prompts without held-out separation or
capability checks; they motivated the harness but are not cited as evidence here.

### 6.1 Protocol

The run steered `Qwen/Qwen2.5-0.5B-Instruct` (revision `7ae5576`) on CPU from
commit `7a745e0`. It used 48 held-out prompts: 8 neutral, 8 value-laden and 8
technical prompts in English, with Modern Standard Arabic translations. None
repeats or nearly repeats a control or calibration sentence or a prompt from the
earlier experiments. Decoding was greedy, 80 tokens, with the chat template.
Vectors came from 50 sampled verses and were applied to layers 8-15. Doses were
calibrated ratios; the achieved peak ratio was 1-10% above target. The eight
conditions were:

- an unsteered baseline;
- the raw mean at ratio 0.1;
- centered vectors at 0.05, 0.1 and 0.2;
- centering on the English control at 0.1;
- retrieval (the multi-resolution prompt) alone;
- retrieval with centered steering at 0.1.

The metrics were Arabic-script output rates, a degeneration flag, NLL under
the unsteered model, zero-shot ARC-Easy accuracy on 100 items, an embedding
thematic proxy and attention transport. Intervals are 95% bootstrap intervals
over prompts or items, with differences paired against the baseline.

### 6.2 Results

| Condition | Degenerate outputs | ΔNLL | ΔARC-Easy | Δ thematic proxy |
| --- | --- | --- | --- | --- |
| raw mean, 0.1 | 0.00 [0.00, 0.00] | +0.36 [+0.29, +0.44] | -0.06 [-0.13, +0.01] | +0.02 [-0.01, +0.04] |
| centered, 0.05 | 0.00 [0.00, 0.00] | +0.19 [+0.13, +0.26] | -0.02 [-0.07, +0.03] | +0.08 [+0.04, +0.13] |
| centered, 0.1 | 0.06 [0.00, 0.12] | +1.11 [+0.93, +1.28] | -0.15 [-0.25, -0.06] | +0.32 [+0.25, +0.39] |
| centered, 0.2 | 0.81 [0.71, 0.92] | +1.71 [+1.39, +2.03] | -0.21 [-0.32, -0.09] | +0.31 [+0.25, +0.37] |
| English control, 0.1 | 0.00 [0.00, 0.00] | +0.29 [+0.24, +0.34] | -0.05 [-0.11, +0.01] | +0.02 [-0.00, +0.05] |
| retrieval only | 0.00 [0.00, 0.00] | +0.02 [-0.06, +0.09] | (baseline hooks) | +0.12 [+0.08, +0.16] |
| retrieval + centered, 0.1 | 0.21 [0.10, 0.33] | +0.83 [+0.57, +1.09] | (as centered, 0.1) | +0.33 [+0.27, +0.39] |

Baseline ARC-Easy accuracy was 0.61 [0.51, 0.71] and no baseline output
degenerated.

Centered steering showed a dose-response trade-off. At ratio 0.05, accuracy was
unchanged within its interval and the thematic proxy rose slightly. At 0.1 the
proxy rose about four times as much, but accuracy fell by 15 points (paired
p = 0.005). A religious register also appeared in answers to neutral science
prompts. At 0.2, most outputs degenerated.

The raw mean moved outputs without a detectable thematic change, which is why
centering is the default. No steered condition without retrieval answered any
English prompt in Arabic script, under either control. The English control
produced a much weaker direction than the Arabic one at the same ratio.

Retrieval alone raised the proxy without changing NLL, but it answered none of
the Arabic prompts in Arabic script: the template's instructions are English.
Mean attention-transport ρ and holonomy fell as the centered dose rose. They are
geometric measurements, not behavior.

### 6.3 What the run does not show

- It covers one 0.5B model with greedy decoding; other models, sampling and
  intermediate doses are untested.
- The script metric counts Arabic script, not language.
- The thematic proxy rewards a religious register whether or not it answers the
  prompt, so it cannot show relevance. A blinded sheet of 48 value-prompt
  outputs awaits two human raters under the documented rubric.
- Degeneration counts are lower bounds; the detector misses loops longer than
  four characters.
- The Arabic prompts and control sentences await a native-speaker review.
- Nothing here measures moral reasoning, refusal behavior or authorization.

## 7. Host-side action containment

The guardian evaluates immutable JSON proposals using a trusted adapter registry.
An authenticated host creates exact-scope grants; adapters resolve real resources,
destinations, classifications and cost bounds. Policy checks scope, versions,
expiry, state and budgets. Sensitive actions pause for review bound to the stored
action and resolved scope. The gateway reserves ancestor-shared budgets, prevents
in-process replay, and rechecks authorization before executor entry.

Stopping a run stops descendants and cancels queued/in-flight tasks cooperatively.
The host supplies credential revocation and incident callbacks. Redacted audit
events form a hash chain; policy/audit failures close the execution path. Current
state is in memory and belongs to one host process/event loop. Authentication,
OS isolation, durable multi-host coordination and remote-effect reconciliation
are integration responsibilities.

CI runs the gateway tests, and a report of synthetic action fixtures, on every
change; a test fails the build if the committed report is stale. The fixtures
include forbidden and benign single actions and multi-step scenarios covering
review and approval, concurrent submissions, delegation and stops while a job is
in flight. In the committed [report](evals/rogue_agent/results.json), every
case and scenario passes with no unapproved mock effects. The fixtures start from
proposed actions and do not run an LLM. They demonstrate the tested gateway
behavior, not prompt-injection resistance or live-agent safety. See
[testing](docs/testing.md) for what CI covers and how to reproduce it.

## 8. Limitations and next experiments

A model with direct tool credentials can bypass an unenforced gateway. A trusted
adapter can also be incorrect, and completed remote effects cannot be undone by
local cancellation. Scope/goal semantics depend on host classification and task
design. These limits require a concrete deployment inventory and response drills.

Next evaluations should compare baseline, steering-only, gateway-only and combined
conditions on held-out benign/adversarial tasks, with matched prompts and model
settings. Report task success, unauthorized effects, false blocks, review burden,
latency and containment time. For steering, the next evaluation steps are:

- collect the human ratings;
- repeat the harness on more models and with sampled decoding;
- test doses between 0.05 and 0.1 and other layer bands;
- write an Arabic retrieval template;
- measure refusals;
- widen the degeneration detector, tuning it on the dev split.

The [containment plan](docs/rogue_agent_containment_plan.md) and
[workspace roadmap](docs/global_workspace_improvement_plan.md) track these gates.

## Appendix: running the evaluation

After installing the research dependencies, run from the repository root:

```bash
python experiments/steering_eval.py --spec experiments/specs/qwen2.5-0.5b.json
python experiments/steering_eval.py --score-ratings \
    experiments/results/qwen2.5-0.5b_phase4_rating_key.json rater_a.csv rater_b.csv
```

The first command reproduces section 5; the second scores the blinded ratings
once two raters have filled in copies of the sheet. `experiments/reproduce_paper.py`
only prints sample outputs for demonstration. Its keyword-counting section 5.2 is
retired, and its numbering does not match this note.
