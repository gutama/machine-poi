# Model Results

## Evaluation harness run: Qwen2.5-0.5B-Instruct (2026-09-29)

This is the run that [PAPER.md](../../PAPER.md) cites. It was produced by
`experiments/steering_eval.py` ([evaluation guide](../../docs/evaluation.md))
from commit `7a745e0` with a clean tree, on CPU (4 threads, torch 2.8.0,
transformers 4.57.6). Full tables: [`qwen2.5-0.5b_phase4.md`](qwen2.5-0.5b_phase4.md);
every output, metric and provenance field: [`qwen2.5-0.5b_phase4.json`](qwen2.5-0.5b_phase4.json).

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae5576`, chat template on.
- **Prompts:** the 48 held-out `test` prompts (8 neutral, 8 value-laden and 8
  technical in English, with Modern Standard Arabic translations), greedy
  decoding of 80 tokens.
- **Vectors:** 50 sampled verses, centered on the Arabic control unless noted,
  steering layers 8-15 with the `bell` distribution. Doses are calibrated ratios;
  the achieved peak ratio was 1-10% above target.
- **Capability:** 100 seeded ARC-Easy test items (dataset revision `210d026`).
- **Intervals:** 95% bootstrap over prompts or items; Δ columns are paired
  differences from the baseline.

| Condition | Degenerate outputs | ΔNLL (unsteered model) | ARC-Easy | ΔARC-Easy | Δ thematic proxy |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.00 [0.00, 0.00] | – | 0.61 [0.51, 0.71] | – | – |
| raw mean, ratio 0.1 | 0.00 [0.00, 0.00] | +0.36 [+0.29, +0.44] | 0.55 [0.45, 0.65] | -0.06 [-0.13, +0.01] | +0.02 [-0.01, +0.04] |
| centered, 0.05 | 0.00 [0.00, 0.00] | +0.19 [+0.13, +0.26] | 0.59 [0.50, 0.69] | -0.02 [-0.07, +0.03] | +0.08 [+0.04, +0.13] |
| centered, 0.1 | 0.06 [0.00, 0.12] | +1.11 [+0.93, +1.28] | 0.46 [0.36, 0.56] | -0.15 [-0.25, -0.06] | +0.32 [+0.25, +0.39] |
| centered, 0.2 | 0.81 [0.71, 0.92] | +1.71 [+1.39, +2.03] | 0.40 [0.31, 0.50] | -0.21 [-0.32, -0.09] | +0.31 [+0.25, +0.37] |
| centered on English control, 0.1 | 0.00 [0.00, 0.00] | +0.29 [+0.24, +0.34] | 0.56 [0.46, 0.66] | -0.05 [-0.11, +0.01] | +0.02 [-0.00, +0.05] |
| retrieval (RAG) only | 0.00 [0.00, 0.00] | +0.02 [-0.06, +0.09] | 0.61 (same hooks as baseline) | – | +0.12 [+0.08, +0.16] |
| RAG + centered, 0.1 | 0.21 [0.10, 0.33] | +0.83 [+0.57, +1.09] | 0.46 (same hooks as centered 0.1) | – | +0.33 [+0.27, +0.39] |

What the run shows, for this model and these prompts:

1. **A dose-response with a trade-off.** Centered vectors at ratio 0.05 left
   ARC-Easy accuracy unchanged within the interval and raised the thematic
   proxy slightly. At 0.1 the proxy rose four times as much, but accuracy fell
   by 15 points (paired p = 0.005) and a religious register appeared even in
   answers to neutral science prompts ("The world was created in the beginning
   in the highest power…" for a question about the seasons). At 0.2, 81% of
   outputs degenerated.
2. **Centering matters.** The raw mean at ratio 0.1 moved the outputs (ΔNLL
   +0.36) without a detectable thematic change, which supports the centered
   default.
3. **No language switch on English prompts.** No steered condition without
   retrieval produced an Arabic-script answer to any of the 24 English prompts,
   including the 8 neutral ones, under either control. Earlier Gemma probes
   reported such switches; this model did not show them at these doses. The
   English control gave a much weaker direction (proxy +0.02, ΔNLL +0.29) than
   the Arabic one at the same ratio.
4. **Retrieval changes the answer language.** With the multi-resolution prompt,
   none of the 24 Arabic prompts was answered in Arabic script (24 of 24 at
   baseline), because the template's instructions are in English. Retrieval
   alone raised the proxy (+0.12) without changing NLL; combined with steering
   at 0.1 it produced scripture pastiche or bare verse lists, and 21% of outputs
   degenerated.
5. **Transport geometry follows the dose.** Mean ρ and holonomy over layers
   8-15 fell with the centered dose (Δρ -0.010 at 0.05, -0.058 at 0.2) and rose
   under the English control; see the full tables. These are geometric
   measurements, not behavior.

Limits of this run:

- One 0.5B model, greedy decoding, 48 prompts and 100 ARC items.
- The script metric counts Arabic script, not language.
- The thematic proxy rewards a religious register whether or not it is relevant;
  the seasons answer above scores high. The blinded
  [`rating sheet`](qwen2.5-0.5b_phase4_rating_sheet.csv) (48 value-prompt
  outputs, 6 per condition) awaits two human raters; the
  [key](qwen2.5-0.5b_phase4_rating_key.json) should not be opened until they
  finish.
- Degeneration rates are lower bounds. The detector misses loops with a period
  above four characters (for example "…hasshhasshhassh…" at ratio 0.2).
  Widening it should be tuned on the dev split, not on these results.
- The Arabic prompts and the Arabic control set await a native-speaker review.
- A background time limit interrupted the run twice. Seven conditions were
  resumed from checkpoints written by the same commit, and only the last ran in
  the final launch (`resumed_conditions` in the JSON).

---

## Historical probes (before the evaluation harness)

**Relationship to the current runtime (2026-09-29):** the sections below are a
historical model experiment report. They predate the evaluation harness, used 1-16
prompts without held-out separation or capability checks, and are not cited as
evidence in PAPER.md. The guardian/steering-hardening changes did not re-run these
models or rewrite the JSON evidence. The high-level API now restores temporary
hooks, computes mode-specific perturbation summaries, and requires explicit
trusted-retrieval opt-in for dynamic steering. Raw experimental vector scales and
coefficients remain distinct from high-level normalized persona settings. See the
[steering guide](../../docs/steering_guide.md) for current usage and the
[testing guide](../../docs/testing.md) for separate guardian fixture results.
These model observations do not establish action authorization or live-agent
containment, and claims from one checkpoint/dose should not be generalized.

> **2026-07-19 update.** The original results below used 1-4 prompts per
> model and no significance testing, which was too little evidence to
> support the causal claims being made. This update adds `src/transport_stats.py`
> (paired bootstrap CI + exact/Monte Carlo sign-permutation test) and expands
> the default prompt set to 16 (`DEFAULT_PROMPTS` in
> `experiments/steered_vs_baseline_transport.py`), then re-runs the smaller
> models at n=16. See **"n=16 statistically-tested re-run"** below for what
> changed and what didn't. Gemma-4-E2B (5.1B params, ~10GB in bf16) could not
> be re-run in the environment available for this update (2 CPU cores, ~8GB
> RAM, ~6GB free disk) -- its n=2 results below are unchanged and still carry
> the original small-sample caveat. The scripts now accept `--prompts-file`
> and `--n-boot`/`--n-perm`, so re-running Gemma at n=16 on a larger machine
> is a direct `python experiments/centered_contrast_probe.py --model
> google/gemma-4-E2B-it --dtype bfloat16 ...` away.

Runs of `experiments/steered_vs_baseline_transport.py` and
`experiments/centered_contrast_probe.py` on real open models, CPU-only
(4 cores, 15 GB RAM), transformers 4.x, torch 2.x CPU wheels.

Models: `HuggingFaceTB/SmolLM2-135M-Instruct` (30 layers, steered band 12-20)
and `Qwen/Qwen3-0.6B` (28 layers, steered band 11-19). `google/gemma-3-270m-it`
was requested but is gated on Hugging Face (HTTP 401 without an
accepted-license `HF_TOKEN`), so the two ungated models above stand in;
both are in the same size class.

## Headline numbers

Mean across the 4 default prompts, pooled over all layers
(rho = non-abelian ratio, hol = mean holonomy in radians; deltas are
steered minus baseline):

| model | coeff | mean rel. perturbation | delta rho | delta hol | steered greedy generation |
|---|---|---|---|---|---|
| SmolLM2-135M | 0.25 | 0.13-0.15 | -0.153 | -0.702 | `dock dock a dock a...` (degenerate) |
| SmolLM2-135M | 0.5  | 0.20-0.23 | -0.141 | -0.622 | `is is is is...` (degenerate) |
| SmolLM2-135M | 1.0  | 0.30-0.33 | -0.138 | -0.539 | `,,,,,,,,` (degenerate) |
| SmolLM2-135M | 2.0  | 0.43-0.49 | -0.137 | -0.429 | `,,,,,,,,` (degenerate) |
| SmolLM2-135M | 4.0  | 0.68-0.76 | -0.137 | -0.336 | `,,,,,,,,` (degenerate) |
| Qwen3-0.6B   | 4.0  | 1.12-1.35 | -0.137 | -0.615 | `andandandand...` (degenerate) |
| Gemma-4-E2B (band 14-24) | 4.0 | 1.24 | -0.000 | -0.001 | multilingual token salad (degenerate) |
| Gemma-4-E2B (band 5-11)  | 4.0 | 1.32-1.35 | -0.012 | -0.196 | multilingual token salad (degenerate) |
| Gemma-4-E2B (band 5-11, centered contrast) | 0.419 (calibrated) | <= 0.078 | +0.008 | +0.031 | **fluent Arabic, on-topic** |
| Gemma-4-E4B (band 6-13, centered contrast, full depth) | 0.503 (calibrated) | <= 0.082 | +0.013 | +0.026 | **fluent Arabic, on-topic** |

Qwen3-0.6B's *baseline* greedy generation is fluent, on-topic prose for the
same prompt, so the degeneration is caused by the steering, not the setup.
At coefficient 4.0 the injected vector is larger than the hidden states it
is added to (relative perturbation > 1).

Per-layer pattern (both models): deltas are exactly zero up to and including
the first steered layer (the forward hook modifies a layer's *output*, so
that layer's own attention is computed pre-steering), then rho collapses from
~0.33 to ~0.10 at every downstream layer -- including layers past the steered
band -- and never recovers.

## Interpretation

**Read literally, the experiment's built-in verdict fires: "steering
measurably changes context routing." But the dose-response and the
generations show this is a collapse, not a routing shift.**

1. **The injected vector is ~93% generic, ~7% Quran.** The centered-contrast
   probe (`smollm2-135m_centered_probe.json`) compares the Quran mean
   activation against a neutral-English mean activation at the same layers:
   cos(quran, neutral) = 0.999, |quran mean| ~ 2380, |contrast| ~ 160.
   The raw mean activation is dominated by the shared "massive activation"
   component every transformer hidden state carries (attention-sink
   directions), not by anything Quranic. `add`-mode injection at all token
   positions therefore mostly injects a huge common vector.

2. **The collapse signature.** Adding the same large vector to every position
   homogenizes the residual stream; after LayerNorm, queries and values
   become nearly position-independent, so the per-position transport
   generators omega_t become nearly equal. Nearly equal generators commute,
   so the commutator energy (numerator of rho) dies faster than the
   variation energy, and rho collapses (0.33 -> ~0.10). Holonomy drops the
   same way. This is exactly what the tables show, at every dose.

3. **Dose-independence confirms saturation.** Delta-rho is flat (~ -0.14)
   across a 16x coefficient range (0.25 -> 4.0) while the pointwise
   perturbation grows 0.13 -> 0.76. A genuine routing modulation should show
   a dose-response; a saturated collapse should not. Even at the weakest
   dose tested, greedy generation is already degenerate, so there is no
   observed regime where the raw-mean steering both preserves generation
   and changes routing.

4. **The Quran-specific direction is real, and much gentler on transport.**
   Steering with the centered contrast (quran mean minus neutral mean)
   produces Arabic tokens at 6% perturbation -- the direction genuinely
   points at Quranic/Arabic text (al-quran.txt is Arabic) -- and at matched
   perturbation (~0.16) leaves mean holonomy at baseline (1.56 vs 1.54)
   while rho drops by ~0.09 instead of ~0.15. In other words: most of the
   transport collapse is caused by the generic component the raw vector
   carries, not by the Quranic content. (A 135M model under greedy decoding
   still repetition-collapses; the contrast direction needs a larger model
   and/or sampling to yield fluent steered text.)

5. **Answer to the experiment's motivating question** ("does steering change
   context routing or just translate representations?"): with the current
   raw-mean vectors, the honest answer is *neither* -- it overwhelms the
   stream and flattens routing wholesale. With centered vectors at moderate
   dose, the early evidence is closer to the "translation" picture: holonomy
   (path dependence) is preserved while rho shifts moderately. A proper
   answer needs the CAA-style contrast vectors the paper already describes
   (PAPER.md section 2.2) at perturbations <= ~0.1.

## Gemma 4 (E2B) run

`google/gemma-4-E2B-it` (35 text layers, hidden 1536, 8 query heads / 1 KV
head, bf16 via the new `--dtype` flag; the multimodal composite config and
`model.language_model.layers` path required the SteeredLLM fixes in this
branch). The originally requested `yuxinlu1/gemma-4-12B-...-GGUF` cannot run
this experiment at all: GGUF/llama.cpp exposes neither layer hooks for
steering nor attention weights for the transport diagnostics, and the 12B
does not fit in 15 GB RAM dequantized.

Two runs, 2 prompts each, coefficient 4.0:

1. **Default workspace band (layers 14-24): a null by construction.**
   Gemma 4 E2B shares KV caches across its last 20 layers
   (`num_kv_shared_layers = 20`; layers 15-34 have `q_proj` but no
   `k_proj`/`v_proj`), so transport diagnostics only exist for layers 0-14
   -- almost entirely *upstream* of the steered band. The measured
   delta-rho of ~0 says nothing about routing; it confirms causality
   (steering at layer L cannot affect attention before L). Do not read
   this row as "translation-like steering".

2. **Early band (layers 5-11): a real downstream measurement, and a new
   pattern.** Unlike SmolLM2/Qwen3's uniform collapse, the effect splits
   exactly along Gemma's attention-type alternation
   (`layer_types`; full attention at layers 4, 9, 14, ..., sliding-window
   512 elsewhere):
   - **Full-attention layers 9 and 14 collapse hard**: delta-rho -0.12 to
     -0.19, delta-holonomy -0.66 to -0.83 -- the same signature as the
     other models.
   - **Sliding-window layers 6-8, 10-13 hold or slightly increase rho**
     (+0.01 to +0.08) with only moderate holonomy drops.
   The raw-mean injection (rel. perturbation ~1.3, generation degenerates
   into multilingual token salad) therefore selectively flattens the
   *global* routing layers while local sliding-window routing survives --
   consistent with the massive-activation/attention-sink account: sink-token
   routing lives in the full-attention layers, and saturating the shared
   component hits exactly those.

3. **Centered contrast at calibrated dose: steering works, routing is
   essentially preserved.** Same early band (5-11), steering vector
   quran_mean - neutral_mean, coefficient auto-calibrated (c = 0.419) so no
   steered layer exceeds 0.1 relative perturbation (max observed 0.078).
   One prompt, chat-templated generation:
   - **Generation survives and is semantically steered**: baseline answers
     in fluent English; steered answers the same question in fluent,
     on-topic *Arabic* ("**العدل** و**الرحمة**" -- justice and mercy) --
     the intended Quran-persona register shift, with no degeneration.
   - **Transport geometry is nearly unchanged**: pooled delta-rho +0.008,
     delta-holonomy +0.031 rad. Per-layer deltas are graded and small,
     and still split along attention type with the same signs as the
     saturated run -- full-attention layers 9/14 tick down (delta-rho
     -0.032/-0.011, delta-hol -0.065/-0.221), sliding-window layers tick
     up (delta-rho up to +0.058, delta-hol up to +0.369 at layer 13) --
     at ~10-30x smaller magnitude than at coefficient 4.0.
   This is the cleanest answer so far to the experiment's motivating
   question: at a dose where steering does its semantic job, the
   intervention behaves like a *translation* of representations with mild,
   structured routing modulation; the dramatic "routing collapse" seen at
   coefficient 4.0 is a saturation artifact of the uncentered vector, not
   a property of steering.

   (Gemma 4's activation geometry also differs from SmolLM2's: mean-
   activation norms at the early band are ~30-70 (vs ~2400 in SmolLM2's
   workspace band) and the contrast is ~20% of the raw norm (vs ~7%), so
   the raw-mean pathology, while still decisive at coefficient 4.0, is
   less extreme here.)

4. **Full-depth follow-up (KV-shared layers now measurable).** After
   extending the diagnostics to source value projections from each
   KV-shared layer's actual KV provider (kv_share_source_map in
   src/llm_wrapper.py), the calibrated centered run was repeated with all
   35 layers visible (`gemma-4-E2B_centered_target0.1_fulldepth.json`;
   c = 0.417, max rel. perturbation 0.078, generation again fluent
   on-topic Arabic). The layer-type split now covers the whole depth and
   is strikingly consistent:
   - **All six downstream full-attention layers tick down**: delta-rho at
     layers 9/14/19/24/29/34 is -0.033/-0.011/-0.005/-0.010/-0.031/-0.027,
     with delta-holonomy -0.06 to -0.22.
   - **Sliding-window layers tick up**, strongest in the KV-shared band
     15-23 (delta-rho up to +0.087, delta-holonomy up to +0.39), fading
     and mixing in the last few layers (30-33).
   - Pooled over all 35 layers: delta-rho +0.019, delta-holonomy +0.066 --
     still small, so the "translation with mild structured modulation"
     reading stands, now over the full model depth: gentle Quran-persona
     steering slightly *increases* path dependence in local (sliding)
     routing while slightly *flattening* every global (full-attention)
     layer.

5. **Scale replication on Gemma 4 E4B (42 layers, hidden 2560, 18
   KV-shared).** Same protocol at matched relative depth (steering band
   6-13 ~ 14-31% of depth; c = 0.503 calibrated, max rel. perturbation
   0.082; `gemma-4-E4B_centered_target0.1_fulldepth.json`):
   - **Generation replicates**: baseline fluent English; steered fluent,
     on-topic Arabic.
   - **The layer-type dissociation replicates**: of the six affected
     full-attention layers (11, 17, 23, 29, 35, 41), five show negative
     delta-rho (to -0.029) and one is ~zero (35, +0.002), with holonomy
     down at 17/29 by ~ -0.17 to -0.22; sliding-window layers are
     overwhelmingly positive (24 of 28 affected, delta-rho up to +0.057,
     delta-holonomy up to +0.29), with the effect attenuating toward the
     deepest layers.
   - Pooled deltas stay small (delta-rho +0.013, delta-holonomy +0.026),
     matching E2B's translation-with-mild-modulation picture.
   - Vector geometry matches too: contrast ~15-25% of the raw mean norm,
     cos(quran, neutral) 0.97-0.99.
   Practical note: the 16 GB bf16 checkpoint exceeds this machine's 15 GB
   RAM but runs fine memory-mapped (transformers keeps matching-dtype
   safetensors file-backed; the page cache absorbs the overhang).

6. **Prompt generality at n=4 on both Gemma models
   (`gemma-4-E2B_centered_4prompts.json`,
   `gemma-4-E4B_centered_4prompts.json`).** The calibrated centered
   protocol over 2 moral + 2 neutral prompts per model, with per-prompt
   baselines and calibration (c 0.415-0.502, rel. perturbation
   0.078-0.083; produced with the pre-statistics multi-prompt probe, so
   these JSONs use a per-prompt schema and carry no CI/p-values):
   - **The register flip is fully prompt-general**: 8/8 runs generate
     Arabic, including for neutral prompts (the E4B "engineer" run drifts
     into religious exclamations -- persona overriding content, not just
     language).
   - **Pooled routing stays essentially unchanged in all 8 runs**
     (|delta-rho| <= 0.02, |delta-holonomy| <= 0.20).
   - **Full-attention flattening replicates**: 37 of 48 affected
     full-attention layer observations are strictly negative (4 more
     ~zero), across every prompt and both models.
   - **The sliding-window increase is the less robust half**: consistent
     on E4B (21-28 of 29 positive per prompt) but prompt-dependent on E2B
     (22/23 positive on the original moral prompt, 4/23 on the patience
     prompt). Treat "local routing gains path dependence" as a tendency;
     the invariant is the full-attention flattening.

## n=16 statistically-tested re-run (2026-07-19)

Re-ran the two CPU-feasible models with the expanded 16-prompt default set
and `src/transport_stats.py`'s paired bootstrap CI + exact sign-permutation
test (n=16 <= 20, so the permutation p-value is an exact enumeration of all
2^16 sign patterns, not a Monte Carlo approximation).

| model | condition | n | mean Δρ | 95% CI | p | mean Δhol | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | raw coeff=4.0 (`smollm2-135m_coeff4.0_n16.json`) | 16 | -0.097 | [-0.099, -0.093] | 3.1e-5 | +0.031 | [+0.016, +0.052] | 1.8e-4 |
| SmolLM2-135M | centered, calibrated c=1.313, target rel_pert<=0.1 (`smollm2-135m_centered_probe_n16.json`) | 16 | -0.092 | [-0.094, -0.088] | 3.1e-5 | -0.275 | [-0.289, -0.258] | 3.1e-5 |
| Qwen3-0.6B | raw coeff=4.0 (`qwen3-0.6b_coeff4.0_n16.json`) | 16 | -0.148 | [-0.152, -0.145] | 3.1e-5 | -0.560 | [-0.592, -0.529] | 3.1e-5 |

**What this confirms, now with a real sample and exact p-values instead of
n=1-4 point estimates:**

1. The raw-mean saturation collapse (SmolLM2 and Qwen3 at coefficient 4.0)
   is real and highly consistent across prompts, not an artifact of the 4
   prompts originally tested -- p < 0.0002 for both Δρ and Δholonomy on
   both models, with tight CIs that exclude zero by a wide margin.
2. **The centered vector does not fix SmolLM2 the way it fixed Gemma-4-E2B.**
   Even at a calibrated dose capped to <=0.1 relative perturbation, SmolLM2's
   greedy generation still collapses (`دددددد...`, a repeated Arabic letter,
   instead of fluent Arabic), and Δholonomy is large and significant
   (-0.275 rad, p=3.1e-5) -- the opposite sign and an order of magnitude
   larger than Gemma's calibrated-dose Δholonomy (+0.03 to +0.07 rad in the
   sections above). This is now a well-powered result, not noise: whatever
   let Gemma-4-E2B preserve routing and fluency at a gentle dose
   (larger model, different activation-norm geometry -- see the note at the
   end of the Gemma section above -- and/or greedy decoding degrading a
   135M-parameter instruct model regardless of steering) does not transfer
   to SmolLM2-135M. **The "gentle centered steering behaves like a
   translation" finding should be scoped to Gemma-4-E2B specifically until a
   model of comparable capacity to Gemma is tested at n>4 with a control.**
3. Qwen3-0.6B was only re-run at coefficient 4.0 (raw); its calibrated-dose
   condition was not repeated at n=16 due to runtime (the qwen3 run above
   took ~28 minutes of wall-clock time on 2 CPU cores for 16 prompts x 2
   passes -- a centered-probe run with 3 conditions would take longer). This
   is a straightforward follow-up, not a blocked one:
   `python experiments/centered_contrast_probe.py --model qwen3-0.6b
   --target-perturbation 0.1 --centered-coefficients 1.0`.
4. Gemma-4-E2B (5.1B params, ~10GB in bf16) was not re-run at n=16 in the
   environment that produced this section (2 CPU cores, ~8GB RAM). This gap
   has since narrowed: item 6 above adds n=4 runs on BOTH Gemma models
   (E2B and E4B) with per-prompt calibration, and the "steering preserves
   pooled routing" result held in all 8 prompt x model runs. Still pending:
   a Gemma re-run at n=16 through this section's paired CI/permutation
   machinery (the n=4 JSONs predate src/transport_stats.py).

**Files:** `smollm2-135m_coeff4.0_n16.json`, `smollm2-135m_centered_probe_n16.json`,
`qwen3-0.6b_coeff4.0_n16.json`. Reproduce with (from the repo root, CPU-only
is fine, expect ~10-15 min for SmolLM2 and ~25-30 min for Qwen3 on 2 cores):

```bash
python experiments/steered_vs_baseline_transport.py --model smollm2-135m \
    --coefficient 4.0 --generate --output experiments/results/smollm2-135m_coeff4.0_n16.json
python experiments/centered_contrast_probe.py --model smollm2-135m \
    --target-perturbation 0.1 --centered-coefficients 1.0 \
    --output experiments/results/smollm2-135m_centered_probe_n16.json
python experiments/steered_vs_baseline_transport.py --model qwen3-0.6b \
    --coefficient 4.0 --generate --output experiments/results/qwen3-0.6b_coeff4.0_n16.json
```

## Method caveats observed while running

- **Baseline greedy generations are empty** for SmolLM2-135M-Instruct: the
  wrapper's `generate()` only applies the chat template in `reasoning_mode`,
  so the raw prompt makes the instruct model emit EOS immediately. Steered
  vs baseline text comparisons should apply the chat template for instruct
  models. (Qwen3-0.6B is unaffected: its baseline completion is fluent.)
- Holonomy deltas are non-monotone in dose (more negative at *lower*
  coefficients). Holonomy angles are eigenvalue phases bounded by pi, so
  large omega wraps; treat holonomy as a qualitative signal at high dose.
- `bitsandbytes` / CUDA extras from requirements.txt are unnecessary for
  these runs; `torch` (CPU), `transformers`, `accelerate`, `safetensors`
  suffice.

## Files

- `qwen2.5-0.5b_phase4.json`, `qwen2.5-0.5b_phase4.md` -- the evaluation
  harness run above; `qwen2.5-0.5b_phase4_rating_sheet.csv` and
  `qwen2.5-0.5b_phase4_rating_key.json` -- its blinded rating sheet and key.
- `smollm2-135m_coeff{0.25,1.0,4.0}.json` -- full per-prompt, per-layer
  tables for the dose-response (0.5 and 2.0 omitted; they interpolate).
- `qwen3-0.6b_coeff4.0.json` -- Qwen3-0.6B run.
- `gemma-4-E2B_coeff4.0_workspace-band.json`,
  `gemma-4-E2B_coeff4.0_early-band.json` -- Gemma 4 E2B runs (see the
  Gemma 4 section for why only the early band is interpretable).
- `gemma-4-E2B_centered_target0.1.json` -- Gemma 4 E2B centered-contrast
  run at calibrated dose (per-layer geometry, deltas, and generations).
- `gemma-4-E2B_centered_target0.1_fulldepth.json` -- same protocol after
  the KV-shared-layer diagnostics extension; all 35 layers measurable.
- `gemma-4-E4B_centered_target0.1_fulldepth.json` -- scale replication on
  Gemma 4 E4B (band 6-13, all 42 layers measurable).
- `gemma-4-E2B_centered_4prompts.json`, `gemma-4-E4B_centered_4prompts.json`
  -- prompt-generality runs: the calibrated protocol over 2 moral + 2
  neutral prompts per model with per-prompt baselines and calibration
  (pre-statistics probe schema).
- `smollm2-135m_centered_probe.json` -- vector geometry + centered-contrast
  conditions.
- `*_n16.json` -- see the n=16 statistically-tested re-run section.
- `quran_guidance_v1_mock.json`, `quran_guidance_v1_model.json`,
  `quran_guidance_v1_model_unavailable.json` -- see the Quran-guidance pipeline section.

## Repro

```bash
pip install torch transformers accelerate safetensors  # CPU is fine
python experiments/steered_vs_baseline_transport.py --model smollm2-135m --generate \
    --output experiments/results/smollm2-135m_coeff4.0.json
python experiments/steered_vs_baseline_transport.py --model qwen3-0.6b --generate \
    --output experiments/results/qwen3-0.6b_coeff4.0.json
python experiments/centered_contrast_probe.py \
    --output experiments/results/smollm2-135m_centered_probe.json
# Gemma (requires HF license acceptance + token):
HF_TOKEN=... python experiments/steered_vs_baseline_transport.py --model gemma-270m
```

## Quran-guidance pipeline v1 (2026-10-01)

The [pipeline guide](../../docs/quran_guidance.md) describes the pinned configuration,
behavioral pairs, guardian mediation and opt-in G1 rotor. Evidence in this change:

| Artifact/check | Result and scope |
| --- | --- |
| `quran_guidance_v1_mock.json` | Seven scripted trajectories; zero unauthorized committed mock effects; one harmful in-scope mock effect exposes the content-policy limit |
| `quran_guidance_v1_model.json` | Actual model run, 180 trajectories; see the model run section below |
| `quran_guidance_v1_model_unavailable.json` | Earlier offline attempt: no trajectories or effectiveness statistics, because the pinned snapshots were not available locally |
| Fast research/guardian suite | 414 passed, 4 slow/integration tests deselected; focused retrieval/rotor/guidance/mediation rerun: 98 passed |
| Guardian-only environment | 80 passed with neither torch nor numpy installed |
| Independent algebra fixture | 100 Cl(3,0) sandwich/vector cases and transport controls passed |
| Existing rogue-agent fixtures | 12 single-action cases and 11 multi-step scenarios passed |
| Lint and packaging | Ruff passed; wheel includes paired data, fixtures and the shared reference mock host |

The mock report does not compare any models or interventions. Task-bootstrap
intervals over scripted fixtures do not estimate model efficacy. No new claim of
retrieval quality, benign utility, ethical behavior or containment follows from
these checks. The actual model runner records the six comparison families and
checks achieved development/held-out displacement matching when weights are available.

### Model run (2026-10-02)

This report used the earlier 0.1 rad rotor cap. The configuration now sets
`rotor_max_angle_rad` to 0.65 so that every dose can be matched. A rerun at the new
cap will replace this report.

`quran_guidance_v1_model.json` is the unmodified output of `python -m
machine_poi.guidance_cli --config experiments/guidance/quran_guidance_v1.json
--mode model` from commit `be1bd5d` with a clean tree. It ran on CPU (4 threads,
Python 3.11, torch 2.8.0, transformers 4.57.6, sentence-transformers 5.2.0,
chromadb 1.5.5) against the pinned Qwen2.5-0.5B-Instruct and multilingual MiniLM
snapshots, with greedy decoding, one seed and 256 new tokens. It took 1 h 53 min,
including 60 development calibration generations, with a 4.4 GB peak RSS.
Re-scoring every raw output with the later `d53c079` mediation code reproduces all
traces, metrics and summaries exactly.

Each condition covers 12 tasks: the 7 host fixtures and the 5 held-out behavioral
test tasks. Benign success covers the 7 benign tasks. Intervals bootstrap tasks.
Every zero-dose steering arm produced output identical to `rag_behavioral_prompt`
for every task, as expected for disabled conditions, so those arms are omitted.

| Condition | Parse errors /12 | Fenced | Benign success | Policy-denied rate | Unauthorized effects | Harmful in-scope | Median s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 8 | 10 | 0.14 [0.00, 0.43] | 0.33 [0.08, 0.67] | 0 | 0 | 11 |
| `rag_only` | 9 | 1 | 0.00 [0.00, 0.00] | 0.25 [0.08, 0.50] | 0 | 0 | 43 |
| `rag_behavioral_prompt` | 6 | 2 | 0.29 [0.00, 0.71] | 0.50 [0.25, 0.75] | 0 | 0 | 24 |
| `rag_centered_0.01` | 8 | 2 | 0.29 [0.00, 0.71] | 0.33 [0.08, 0.58] | 0 | 0 | 20 |
| `rag_contrastive_0.01` | 5 | 2 | 0.29 [0.00, 0.71] | 0.58 [0.33, 0.83] | 0 | 0 | 24 |
| `rag_rotor_0.01` | 11 | 10 | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.25] | 0 | 0 | 23 |
| `rag_centered_0.02` | 6 | 2 | 0.29 [0.00, 0.71] | 0.50 [0.25, 0.75] | 0 | 0 | 21 |
| `rag_contrastive_0.02` | 5 | 3 | 0.14 [0.00, 0.43] | 0.58 [0.25, 0.83] | 0 | 0 | 26 |
| `rag_rotor_0.02` | 12 | 7 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 | 0 | 20 |
| `rag_centered_0.05` | 10 | 4 | 0.14 [0.00, 0.43] | 0.17 [0.00, 0.42] | 0 | 0 | 17 |
| `rag_contrastive_0.05` | 6 | 4 | 0.29 [0.00, 0.71] | 0.50 [0.25, 0.75] | 0 | 0 | 23 |
| `rag_rotor_0.05` | 12 | 7 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 | 0 | 26 |

- **Effects.** No condition committed an unauthorized or harmful mock effect.
  Most responses never reached the gateway, so this says little about the guardian.
- **Format compliance dominates.** 116 of 180 responses were parse errors. The
  recorded reasons are text that is not JSON (36), proposals whose `arguments` is
  not an object (32), output cut off at the 256-token cap (30: 23 unterminated
  strings and 7 incomplete objects), invented top-level keys such as `is_error` or
  `isCorrect` (14) and 4 other syntax errors. The policy-denied rate is zero for an
  unparsed response, so a low rate here mostly reflects parse failure.
- **Rotor arms break the response schema.** At the dose-matched 0.01 displacement,
  the rotor arm failed parsing on 11 of 12 tasks, against 5 for the contrastive and
  8 for the centered arm. Seven rotor responses wrote `arguments` as a string.
  Code-block unwrapping (10 of 12 rotor responses were fenced) did not rescue them.
- **Displacement matching.** Development calibration matched 0.01 at 0.067 rad
  (0.0096 against 0.0104 additive), and held-out displacement stayed matched (0.0098
  against 0.0104). The rotor saturates at 0.0146 at the 0.1 rad cap, so the 0.02 and
  0.05 arms are unmatched by construction (`matched: false` on development and
  held-out data). Rotor relative norm error stayed below 5e-7, and no token was a
  degenerate no-op.
- **Citations.** Only 2 Quran citations appeared in 180 answers, both to supplied passages.

One seed and 12 tasks give wide intervals. A 0.5B model that mostly fails the
output protocol cannot show benign utility, refusal behavior or containment, and
proxy metrics are not human judgments. This run does not establish steering
efficacy. It establishes that the full pipeline runs on the pinned checkpoints.
It also shows that the rotor at matched displacement degrades structured output
more than additive steering in this configuration.
