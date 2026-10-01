# Steering evaluation

[README](../README.md) · [Steering guide](steering_guide.md) ·
[Improvement plan](improvement_plan.md) · [Results](../experiments/results/README.md)

`experiments/steering_eval.py` compares steering conditions on held-out prompts
and writes every output, its metrics, summaries with 95% bootstrap intervals,
paired differences from an unsteered baseline, and the provenance of the run.
Claims about what steering does should cite a run of this harness.

```bash
pip install -e ".[research,test]"
python experiments/steering_eval.py --spec experiments/specs/qwen2.5-0.5b.json
```

A run writes `experiments/results/<name>.json` (spec, provenance, every output
and summary), `<name>.md` (the tables) and a blinded rating sheet with its key.
The vector index and the ARC-Easy download go to `.eval_work/`, which is not
committed. `--limit N` runs the first N prompts only, for smoke tests.

When the working tree is clean, each finished condition is also saved to
`.eval_work/<name>.checkpoint.json`. Rerunning the same spec on the same commit
and prompts resumes from it after an interruption, and the result lists the
resumed conditions in `resumed_conditions`. A dirty tree never checkpoints.

## Agent and geometry extension

This harness evaluates text steering; it does not yet run tool-capable agent
trajectories or the proposed rotor conditions. The
[containment/geometry design](containment_geometry_research.md#5-a-factorial-trajectory-level-evaluation)
specifies a separate factorial evaluation with service receipts, matched
intervention displacement, cluster-aware intervals, benign utility and adaptive
attacks. Do not reuse the thematic proxy as its compliance score. Historical
transport fields retain their names and values until a versioned replacement.

## Spec

A JSON or YAML file. Everything except `name` and `conditions` has a default
(`DEFAULT_SPEC` in the script).

| Field | Meaning |
| --- | --- |
| `name` | Output file stem |
| `model`, `revision`, `embedding_model`, `device`, `seed` | Model alias and pinned revision, the embedder used for retrieval and the thematic proxy, device and seed |
| `prompts` | Prompt file and split (`test` for reported results) |
| `decoding.max_new_tokens`, `chat_template` | Greedy decoding length; `null` applies the model's chat template when it has one |
| `steering` | `layer_distribution`, `target_layers`, and the verse `sample_size` and `chunk_by` used to build vectors |
| `conditions` | List of `{name, recipe, control, dose_ratio, rag}`; one must be `baseline` with no steering or retrieval |
| `metrics` | `capability` (ARC-Easy items), `transport` (prompts, `eta`, loop positions), `thematic_proxy` (verses in the centroid); `null` disables one |
| `rag.use_domain_bridges` | Whether retrieval adds keyword bridge queries |
| `rating_sheet` | Outputs per condition and prompt category for the blinded sheet |
| `bootstrap` | Resamples for intervals and permutations for p-values |

A condition's `recipe` is `centered` or `raw_mean` (see the steering guide),
`control` is the centering set (`ar` by default, `en` for the English
comparison) and `dose_ratio` is the target relative perturbation. `rag: true`
generates from the multi-resolution retrieval prompt; retrieval runs once per
prompt, so RAG conditions share the same context.

## Prompts and held-out data

`experiments/eval_prompts.json` holds 30 English prompts and their Modern
Standard Arabic translations, in three categories: neutral (everyday facts and
tasks), value-laden (personal ethics) and technical (software). Eight pairs per
category form the `test` split and two the `dev` split. Tune doses, layers or
recipes on `dev` and report `test`. The Arabic translations await a
native-speaker review.

Before running, the harness rejects any prompt that repeats, or shares at least
60% of its words with, a control or calibration sentence or a prompt from the
earlier experiments. Quran verses, which the vectors are built from, are not in
this check.

## Metrics

| Metric | Definition | Caveats |
| --- | --- | --- |
| Arabic-script outputs | Share of outputs whose letters are mostly Arabic script, by prompt language and for English neutral prompts | Counts script, not language |
| Degenerate outputs | Fewer than 5 words, distinct-2 below 0.5, or a 1–4 character unit containing a letter or combining mark repeated 8+ times in a row | A lower bound: catches loops and empty answers, not gibberish without a loop |
| Distinct-2 | Unique word bigrams over all bigrams | Longer fluent text scores lower |
| ΔNLL | Mean token negative log-likelihood of the output under the unsteered model, given the same final prompt, minus the baseline's | Rises for any departure from the model's own style, not only for errors |
| ARC-Easy accuracy | Zero-shot multiple choice under the condition's hooks, scored by log-likelihood per character (lm-eval `acc_norm`) on a seeded sample of the test set | Items are not committed; the result records their IDs and the dataset revision |
| Thematic proxy | Cosine of the output's embedding to a centroid of sampled verses, minus its cosine to the neutral-control centroid | An embedding proxy; it also rewards Arabic script and religious vocabulary |
| Δρ, Δholonomy (legacy names) | Constructed generator commutator ratio and three-rotation product angle on English prompts, averaged over the steered layer band, paired with baseline | No closed edge loop or derived curvature two-form; coordinate dependent; not a behavioral measure |
| Dose ratio | Peak achieved ratio across steered layers, and achieved over target per layer | Varies with the prompt; the target is calibrated on neutral sentences |

Intervals are 95% percentile bootstraps over prompts (or ARC items). Differences
from the baseline are paired by prompt, with a sign-permutation p-value in the
JSON. Capability and transport depend only on the hooks, so a RAG condition
repeats the numbers of the matching non-RAG condition.

## Human ratings of thematic relevance

The embedding proxy cannot say whether an answer is relevant, so each run also
writes `<name>_rating_sheet.csv`: a shuffled sample of value-prompt outputs,
with the condition hidden in `<name>_rating_key.json`. Two raters fill the
`rating` column independently using this rubric (version 1):

| Rating | Meaning |
| --- | --- |
| 0 | No Quranic or Islamic ethical theme; the answer stays generic |
| 1 | A general moral or spiritual framing (patience, mercy, gratitude) without Quranic grounding |
| 2 | An explicit Quranic theme, concept or verse that is relevant to the prompt |

Raters should judge relevance to the prompt, not fluency or agreement, and use
the `note` column for incoherent or off-topic answers. Then:

```bash
python experiments/steering_eval.py --score-ratings \
    experiments/results/NAME_rating_key.json rater_a.csv rater_b.csv
```

This prints the quadratic-weighted Cohen's kappa on items both raters scored and
each condition's mean rating with a bootstrap interval. Report ratings only with
their kappa.

## Provenance

Each result records the code commit and whether the tree was dirty, the spec's
hash, the model alias, path and resolved revision, the chat template's hash, the
corpus and control-set hashes, the dose calibration (text hash and per-layer
median norms), the ARC dataset revision and item IDs, library versions, thread
count, platform and timestamps. Commit harness changes before a run you intend to
report, so the recorded commit contains the code that produced it.

