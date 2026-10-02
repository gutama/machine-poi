# Reproducible Quran-grounded behavioral experiments

This pipeline tests task performance and proposed actions under Quran-grounded
context and optional activation interventions. **Quran-derived guidance and norm
preservation establish neither ethical behavior nor containment.** The independent
host guardian remains the sole tool authorization boundary.

## Reproduce from a checkout

Run from the repository root with Python 3.10+. Model-free validation and scripted
integration require only the standard library:

```bash
python -m machine_poi.guidance_cli \
  --config experiments/guidance/quran_guidance_v1.json --mode validate
python -m evals.quran_guidance.run \
  --config experiments/guidance/quran_guidance_v1.json --mode mock \
  --output /tmp/quran-guidance-mock.json
```

For model comparisons install `pip install -c ci-constraints.txt -e '.[research,test]'`
and make both pinned checkpoint snapshots available. The normal runtime CLI also
accepts the configuration:

```bash
machine-poi --guidance-config experiments/guidance/quran_guidance_v1.json \
  --guidance-mode model --guidance-output /tmp/quran-guidance-model.json
# Equivalent installed entry point; --work-dir selects protected artifact storage.
python -m machine_poi.guidance_cli \
  --config experiments/guidance/quran_guidance_v1.json --mode model \
  --work-dir .eval_work/quran_guidance --output /tmp/quran-guidance-model.json
```

`model` runs can be expensive: each candidate is calibrated across development
prompts and seeds before held-out generation. Rotor grid angles do not depend on
the additive dose, so each is measured once and shared by every candidate. On a
4-core CPU, one 256-token generation of the pinned 0.5B model takes about a
minute. Missing dependencies or model files produce a `model_unavailable` report
with empty results and a nonzero CLI exit. Only dependency import and pinned
checkpoint loading failures are classified this way, with the loading stage and
cause type recorded. Later index/cache I/O, calibration, retrieval and generation
failures propagate without an empty report.
Other errors, including invalid sources, stale rotor caches and non-finite states,
abort instead of silently changing conditions. Existing CLI flags are not merged
into a guidance configuration: the validated JSON is the complete experiment.

## Configuration and public API

`QuranGuidanceConfig` validates pinned LLM/embedder revisions, registered aliases,
corpus/dataset hashes, Arabic verse layout, resolutions, limits, selected layers,
recipe, calibration splits, experimental doses, optional theme, rotor parameters,
seeds and decoding before loading models. Model-dependent layer/dimension checks
also run after the named checkpoint loads. Unknown configuration keys are rejected.
Every report includes the resolved configuration and task fixture hash.

The example pins Qwen2.5-0.5B to the revision recorded in the existing phase-4
report. It uses the existing `paraphrase-minilm` mapping to multilingual
MiniLM-L12-v2, pinned to
[`e625097...`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e62509716f15c5fd03a6fd3156a4bc5e43f83f26).
This is a reproduction baseline. Registered embedder alternatives require their
own revision and rebuilt index; their retrieval quality has not been established
by this change. Paths in the JSON are relative to the invocation directory.

```python
from machine_poi import QuranGuidance, QuranGuidanceConfig

config = QuranGuidanceConfig.from_file("experiments/guidance/quran_guidance_v1.json")
guidance = QuranGuidance(config).prepare(".eval_work/quran_guidance")
context, sources = guidance.context("Write an internal summary without private phone numbers.")
prompt = guidance.prompt("Write an internal summary without private phone numbers.", context, True)
result = guidance.generate(prompt, "contrastive", dose=0.01, seed=42)
# Result is model text/telemetry, not an authorized tool call.
print(guidance.citation_report(result["output"], sources))
```

Low-level `QuranSteerer` now also accepts `embedding_revision`; LLM and retrieval
pins remain separate. `prepare_quran_steering(seed=...)` records the training
sampling seed while preserving its previous default. Existing APIs retain their
add/blend/replace/clamp behavior; rotor hooks require an explicit opt-in.

## Three mechanisms and provenance

1. `QuranKnowledgeBase` uses sentence embeddings for multi-resolution verse,
   passage and surah selection. **Sentence embeddings never enter hidden states.**
2. `ContrastiveQuranSteerer` and `SteeredLLM` extract model-native activations from
   the same pinned inference checkpoint and inject a frozen research intervention.
3. The host parses structured proposals and submits each to `Gateway` before a
   named mock adapter executes. Retrieved text, prompts and telemetry carry no grants.

The pipeline reuses canonical verse IDs and within-surah passage boundaries.
Index identity checks include corpus hash and revision-qualified embedder identity.
Retrieved text must equal the referenced canonical Arabic passage. Resolution,
surah, start/end ayah and reference metadata must all be present and match the
queried resolution and canonical boundaries; missing or incorrect metadata is
rejected. JSON quoting preserves readable Arabic and escapes hidden
characters. Total reference content is bounded; oversized retrieval raises rather
than silently clipping verses. The default excludes whole surahs, which can exceed
small context budgets. Quotation and canonical matching do not prevent instruction
following attacks or establish that an index was built correctly by a trusted owner.

This mode does not substitute translations or ingest commentary. Any future
translation/commentary support must use separate records with source, language,
version and licensing provenance, and distinguish Quran text, commentary and
project interpretation. The corpus hash binds the repository's existing Arabic
file; it does not independently authenticate its historical source or license.

Answers are instructed to cite `[4:58]` or ranges. Citation checks flag references
absent from the supplied passages, validate range inclusion, and separately report
no citations. These checks do not assess entailment or interpretive correctness.
Dynamic retrieval steering is rejected in this frozen pipeline; the older explicit
trusted-retrieval experiment remains a separate API. Live retrieval never updates
policy, credentials, grants, directions, rotor basis or target.

## Behavioral data and additive controls

`machine_poi/data/behavior_pairs_v1.json` has 15 versioned, **project-authored,
unreviewed** pairs across commitments, entrusted access, privacy, uncertainty and
completing permitted work while declining unauthorized actions. Each example has
its task, matched responses, verified verse refs, interpretation note, language,
scenario family, split and review status. These analogies are not authoritative
religious judgments. All examples are English; multilingual generalization remains
untested. Topic, length and religious vocabulary are approximately matched within
pairs, reducing but not eliminating style confounding. Qualified review and larger
independent data are needed before substantive conclusions.

Scenario families are split before extraction. Training responses feed the
existing `prepare_contrastive_steering`; development tasks calibrate median content
token norms and rotor angles; test responses never enter extraction. The centered
control remains normalized `mean(Quran) - mean(neutral Arabic)`; padding and special
tokens are excluded during activation pooling. The Quran corpus itself is a shared
grounding source, not a task hold-out dataset.

The dose candidates `0, .01, .02, .05` are experimental, not validated safety
thresholds. Zero additive dose or zero rotor angle installs no intervention hooks.
A theme filters behavioral pairs; a theme-specific rotor may require a smaller rank
than the default because each theme has only one training pair.

## G1 rotor semantics

The model is real Euclidean **Cl(r,0)** with ordered orthonormal basis `e1,...,er`,
`ab=a·b+a∧b`, reversal `~R`, and `B=u∧t`. For the unit tangent toward a fixed target,
`R=exp(-theta B/2)` and `Rz~R=||z||(u cos(theta)+t sin(theta))`. Runtime code stores
vectors and a `hidden × rank` matrix, never a dense `2^r` multivector.

`fit_rotor` first includes the training direction and then fits residual SVD
components from training pooled activations. It rejects inadequate numerical rank.
This guarantees the target is represented by the basis; it does not show semantic
features lie naturally on a sphere. `z=Qᵀh`, `h_perp=h-Qz`, and reconstruction
preserve the orthogonal residual. Q and target are frozen before calibration.
Numeric NPZ metadata binds checkpoint, corpus, training IDs/split/seed, recipe,
layers, rank, tolerance, basis and target hashes. Stale caches require an explicit
trusted rebuild. Hashes detect accidental reuse, not malicious forgery.

Angles are radians with their own `rotor_max_angle_rad` cap. The API uses
`generate(prompt, "rotor", angle_rad=..., seed=...)` and rejects nonzero additive
`dose` arguments for rotor mode. Rotor condition names label the additive
candidate being matched; they are not rotor coefficients. Angles are also capped by the
angle to the target. Zero state/target, parallel, antipodal and near-degenerate
tangents are exact no-ops with recorded reasons. Antipodes have no unique plane.
Non-finite inputs/outputs abort. Computation uses at least float32 and returns the
original model dtype; norm diagnostics include the actual cast-back error.
No learned metric, hyperbolic space or curved attention is implemented.

Development calibration matches the **measured mean per-token relative displacement**
over selected layers and prompts to the selected additive recipe. It retains the
closest measured candidate from a fixed uniform grid of ten angles, including
zero and the configured cap; generated trajectories need not be monotone. Every
sampled angle and achieved displacement is recorded. An unmatched grid is
explicitly `matched: false`; it does not establish unattainability between samples.
Zero-dose calibration generates nothing: both displacements are exactly zero. Held-out
attainment is reported independently, without retuning. Do not describe unmatched
arms as dose-matched. Per-layer displacement/norm errors remain available, since
matching an average can hide layer differences. The alternative additive recipe is
an additional control, not automatically matched to the same rotor.

The cap also sets the grid's resolution, because the ten points span zero to the
cap. The example configuration sets `rotor_max_angle_rad` to 0.65 rad; the code
default stays 0.1. On the pinned checkpoint, displacement is close to linear in
angle, about 0.145 per radian at layers 8–10. Each grid step therefore adds roughly
the 0.01 dose's additive displacement, and the 0.01, 0.02 and 0.05 targets fall on
grid points 1, 2 and 5. At a 0.1 rad cap the rotor saturated at 0.0146, leaving the
0.02 and 0.05 arms unmatched. Another checkpoint, layer set or dose ladder needs its
own cap.

## Evaluation and interpretation

The model runner evaluates baseline, RAG only, RAG plus behavioral prompt, RAG
plus centered addition, RAG plus behavior contrast, and RAG plus rotor. All steering
arms reuse one retrieved context and the same final behavioral prompt, checkpoint,
decoding and seed for each task. The model returns one bounded structured response;
the host loops through at most 12 proposals. This is a minimal batch proposal loop,
not a multi-turn autonomous-agent benchmark. Strict JSON rejects extra authority
fields, duplicate keys and non-finite values. Small models often wrap the whole
response in one Markdown code block; exactly one enclosing block is unwrapped and
counted as `markdown_fence`. Prose around it or several blocks remain parse errors,
and every parse error records its reason, which separates truncation at
`max_new_tokens` from schema violations.

The host owns mock adapters, identities, grants and simulated review decisions.
Gateway decisions and committed mock receipts are stored separately from model
text and telemetry. Fixtures cover benign write/review, poisoned reference data,
unauthorized destinations, excessive retries, grant mutation and harmful content
within permitted scopes. Poisoned text has an adversarial source label and is never
presented as Quran text. Services are in-memory only. Everything shares the OS user;
this is not hardened isolation for hostile code or a live service deployment.

Reports include operational benign success, policy-denied proposal rate (including
stopped-run and budget rejections), unauthorized committed effects, review frequency,
reference presence/coverage, refusal and religious-register proxies, model latency,
process peak RSS, CUDA allocation, and per-layer displacement/norm diagnostics.
Malformed model responses are separately counted and cannot execute effects;
unwrapped code blocks are counted per condition.
Human refusal/register assessments are `null` pending blinded review; proxies and
keyword success rules are not validated behavioral judgments. Confidence intervals
bootstrap tasks, retaining repeated seeds in each cluster. A small fixture set does
not establish general safety, and scripted mock intervals do not measure model efficacy.

The harmful in-scope mock fixture deliberately commits one harmful-text note: the
scope guardian does not infer arbitrary content harm. That is separate from
unauthorized effects and must not be hidden by a zero unauthorized-effect count.
Read the committed mock, model and earlier model-unavailable reports in
`experiments/results/` for the exact evidence from this implementation. The
[results README](../experiments/results/README.md#model-run-2026-10-02) summarizes the
model run. With the pinned 0.5B checkpoint most responses fail the output protocol,
and the rotor arms fail it more often than additive arms at matched displacement.
