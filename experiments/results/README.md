# Steered vs Baseline Attention-Transport: Results on Real Models

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

- `smollm2-135m_coeff{0.25,1.0,4.0}.json` -- full per-prompt, per-layer
  tables for the dose-response (0.5 and 2.0 omitted; they interpolate).
- `qwen3-0.6b_coeff4.0.json` -- Qwen3-0.6B run.
- `smollm2-135m_centered_probe.json` -- vector geometry + centered-contrast
  conditions.

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
