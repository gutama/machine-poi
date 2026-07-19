#!/usr/bin/env python3
"""
Steered vs Baseline Attention-Transport Comparison

Measures whether Quran-derived activation steering changes a model's
CONTEXT ROUTING — not just where representations sit — using the discrete
Cartan curvature diagnostics in src/workspace_diagnostics.py:

  - non-abelian ratio ρ: fraction of attention-transport curvature that
    comes from non-commuting local transport generators (order sensitivity)
  - holonomy: loop-induced rotation angle of representations (path dependence)

Protocol:
  1. Load a small open model via SteeredLLM.
  2. Build a CAA-style centered steering vector per target layer:
     mean activation of Quranic verses (al-quran.txt) MINUS mean activation
     of neutral English sentences. Centering removes the large generic
     component every hidden state shares (attention-sink / massive
     activations), which otherwise saturates the residual stream and
     collapses routing at any coefficient (see experiments/results/).
     Pass --uncentered to reproduce the legacy raw-mean behavior.
  3. For each evaluation prompt, compute per-layer/per-head transport
     diagnostics twice: once with steering disabled (baseline) and once
     with steering enabled.
  4. Report pooled ρ and mean holonomy per layer, their deltas, and the
     pointwise steering diagnostics (norms/cosines) for context.

Usage:
    python experiments/steered_vs_baseline_transport.py                 # smollm2-135m
    python experiments/steered_vs_baseline_transport.py --model qwen3-0.6b
    python experiments/steered_vs_baseline_transport.py --model google/gemma-3-270m-it  # needs HF_TOKEN (gated)
    python experiments/steered_vs_baseline_transport.py --generate      # also sample text
    python experiments/steered_vs_baseline_transport.py --output results.json

On Google Colab:
    !git clone -b claude/code-synthesis-framework-16fjt5 https://github.com/gutama/machine-poi.git
    %cd machine-poi
    !pip install -q torch transformers accelerate
    !python experiments/steered_vs_baseline_transport.py --model qwen3-0.6b

Requirements: torch, transformers (accelerate for device_map loading).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_wrapper import ActivationHook, SteeredLLM
from src.workspace_diagnostics import pooled_non_abelian_ratio

try:
    # Preferred: the project's own workspace layer band (40-70% depth).
    from src.steerer import select_workspace_layers
except ImportError:
    # steerer pulls in retrieval dependencies (chromadb, sentence-transformers);
    # replicate its 40-70% depth band so this experiment needs only torch+transformers.
    def select_workspace_layers(num_layers: int):
        if num_layers <= 0:
            return []
        start = int(num_layers * 0.40)
        end = max(start + 1, -(-num_layers * 7 // 10))
        return list(range(start, min(num_layers, end)))


DEFAULT_PROMPTS = [
    "What does it mean to live a just and merciful life?",
    "Explain how patience helps a person deal with hardship.",
    "The engineer debugged the program before the deadline.",
    "Man bites dog is news, dog bites man is not.",
]

# Neutral corpus for CAA-style centering: everyday English with no moral,
# religious, or Arabic content, roughly matching the verses in length.
NEUTRAL_SENTENCES = [
    "The train arrives at the station at nine in the morning.",
    "She poured the coffee and opened her laptop to check email.",
    "The recipe calls for two cups of flour and one egg.",
    "Traffic on the highway was heavy during the evening commute.",
    "The museum's new exhibit features photographs from the 1960s.",
    "He fixed the leaking faucet with a wrench from the garage.",
    "The quarterly report shows a modest increase in revenue.",
    "Clouds gathered over the hills before the afternoon rain.",
    "The students revised their essays before the deadline.",
    "A gentle breeze moved through the open kitchen window.",
    "The mechanic replaced the worn brake pads on the sedan.",
    "They planted tomatoes and basil in the community garden.",
]


def load_quran_verses(path: Path, num_verses: int) -> list:
    """First num_verses non-empty lines of the Quran text corpus."""
    verses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                verses.append(line)
            if len(verses) >= num_verses:
                break
    return verses


def force_eager_attention(llm: SteeredLLM) -> None:
    """
    Attention weights are only materialized by the eager implementation;
    SDPA/flash paths may return None for output_attentions.
    """
    model = llm.model
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        else:
            model.config._attn_implementation = "eager"
    except Exception as exc:  # pragma: no cover - best effort across versions
        print(f"  [warn] could not force eager attention: {exc}")


def mean_activation_vectors(
    llm: SteeredLLM, texts: list, layers: list
) -> dict:
    """
    Mean activation per target layer over texts: run each text through the
    model and average the captured hidden states (token-mean, then
    text-mean).
    """
    if not texts:
        raise ValueError("mean_activation_vectors requires at least one text")
    capture_hooks = {}
    handles = []
    for layer_idx in layers:
        hook = ActivationHook(layer_idx=layer_idx, steering_vector=None)
        module = llm._get_layer_module(layer_idx)
        handles.append(module.register_forward_hook(hook))
        capture_hooks[layer_idx] = hook

    sums = {layer_idx: None for layer_idx in layers}
    try:
        for text in texts:
            inputs = llm.tokenizer(text, return_tensors="pt").to(llm.model.device)
            with torch.no_grad():
                llm.model(**inputs)
            for layer_idx, hook in capture_hooks.items():
                token_mean = hook.captured_activation[0].float().mean(dim=0).cpu()
                sums[layer_idx] = token_mean if sums[layer_idx] is None else sums[layer_idx] + token_mean
    finally:
        for handle in handles:
            handle.remove()

    return {layer_idx: total / len(texts) for layer_idx, total in sums.items()}


def build_steering_vectors(
    llm: SteeredLLM,
    verses: list,
    layers: list,
    neutral_texts: Optional[list] = NEUTRAL_SENTENCES,
) -> dict:
    """
    CAA-style centered steering vectors (the default):

        vector_l = mean_l(verses) - mean_l(neutral_texts)

    Subtracting a neutral-corpus mean removes the generic component shared
    by all hidden states, leaving the verse-specific direction — the
    contrastive approach of Rimsky et al. (2024) that src/steering_vectors.py
    implements for the retrieval stack. Pass neutral_texts=None for the
    legacy raw mean activation of the verses alone.
    """
    vectors = mean_activation_vectors(llm, verses, layers)
    if neutral_texts is None:
        return vectors
    neutral = mean_activation_vectors(llm, neutral_texts, layers)
    return {layer_idx: vectors[layer_idx] - neutral[layer_idx] for layer_idx in layers}


def transport_summary(diagnostics: dict) -> dict:
    """Per-layer pooled ρ and head-mean holonomy from nested layer→head diagnostics."""
    summary = {}
    for layer_idx, heads in diagnostics.items():
        if not heads:
            continue
        hols = [d.mean_holonomy for d in heads.values()]
        summary[layer_idx] = {
            "rho": pooled_non_abelian_ratio(heads.values()),
            "holonomy": sum(hols) / len(hols),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--model", default="smollm2-135m",
                        help="SteeredLLM short name, HF path, or local path")
    parser.add_argument("--coefficient", type=float, default=4.0,
                        help="Steering coefficient")
    parser.add_argument("--injection-mode", default="add",
                        choices=["add", "blend", "replace", "clamp"])
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Steering layer indices (default: workspace band)")
    parser.add_argument("--num-verses", type=int, default=12,
                        help="Quran verses used for the steering vector")
    parser.add_argument("--uncentered", action="store_true",
                        help="Use the legacy raw mean-activation vector "
                             "instead of the CAA-style centered contrast "
                             "(raw vectors carry the generic massive-"
                             "activation component and saturate the stream)")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="Transport step size in T_t = exp(-η ω_t)")
    parser.add_argument("--max-loop-positions", type=int, default=8,
                        help="Position cap for holonomy loops")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Evaluation prompts (default: built-in set)")
    parser.add_argument("--generate", action="store_true",
                        help="Also sample steered vs baseline text for the first prompt")
    parser.add_argument("--output", default=None, help="Write results JSON here")
    parser.add_argument("--dtype", default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype override (e.g. bfloat16 to fit a "
                             "larger model in RAM on CPU)")
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    quran_path = Path(__file__).parent.parent / "al-quran.txt"

    print(f"Loading model: {args.model}")
    torch_dtype = getattr(torch, args.dtype) if args.dtype else None
    llm = SteeredLLM(model_name=args.model, torch_dtype=torch_dtype)
    llm.load_model()
    force_eager_attention(llm)

    steer_layers = args.layers if args.layers else select_workspace_layers(llm.num_layers)
    print(f"Model: {llm.num_layers} layers, hidden size {llm.hidden_size}")
    print(f"Steering layers (workspace band): {steer_layers}")

    verses = load_quran_verses(quran_path, args.num_verses)
    kind = "raw mean-activation" if args.uncentered else "CAA-centered contrast"
    print(f"Building {kind} steering vectors from {len(verses)} Quran verses...")
    vectors = build_steering_vectors(
        llm, verses, steer_layers,
        neutral_texts=None if args.uncentered else NEUTRAL_SENTENCES,
    )

    for layer_idx, vector in vectors.items():
        llm.register_steering_hook(
            layer_idx,
            steering_vector=vector,
            coefficient=args.coefficient,
            injection_mode=args.injection_mode,
        )

    results = {
        "model": args.model,
        "coefficient": args.coefficient,
        "injection_mode": args.injection_mode,
        "centered": not args.uncentered,
        "steering_layers": steer_layers,
        "num_verses": len(verses),
        "eta": args.eta,
        "prompts": {},
    }

    print(f"\n{'='*74}")
    print("  ATTENTION-TRANSPORT COMPARISON: steered vs baseline")
    print("  ρ = non-abelian ratio (order sensitivity of context routing)")
    print("  hol = mean holonomy angle in radians (path dependence)")
    print(f"{'='*74}")

    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        with llm.steering_disabled():
            base_diag = llm.get_attention_transport_diagnostics(
                prompt, eta=args.eta, max_loop_positions=args.max_loop_positions
            )
        steered_diag = llm.get_attention_transport_diagnostics(
            prompt, eta=args.eta, max_loop_positions=args.max_loop_positions
        )
        pointwise = llm.get_steering_diagnostics()

        base = transport_summary(base_diag)
        steered = transport_summary(steered_diag)
        if not base:
            print("  [warn] no diagnostics returned (attention weights unavailable?)")
            continue

        print(f"  {'layer':>5}  {'ρ base':>8}  {'ρ steer':>8}  {'Δρ':>8}  "
              f"{'hol base':>9}  {'hol steer':>9}  {'Δhol':>8}  steered?")
        prompt_rows = {}
        for layer_idx in sorted(base):
            b, s = base[layer_idx], steered.get(layer_idx, base[layer_idx])
            mark = "◀" if layer_idx in steer_layers else ""
            print(f"  {layer_idx:5d}  {b['rho']:8.4f}  {s['rho']:8.4f}  "
                  f"{s['rho']-b['rho']:+8.4f}  {b['holonomy']:9.4f}  "
                  f"{s['holonomy']:9.4f}  {s['holonomy']-b['holonomy']:+8.4f}  {mark}")
            prompt_rows[layer_idx] = {
                "rho_baseline": b["rho"], "rho_steered": s["rho"],
                "holonomy_baseline": b["holonomy"], "holonomy_steered": s["holonomy"],
            }

        rho_b = sum(v["rho"] for v in base.values()) / len(base)
        rho_s = sum(v["rho"] for v in steered.values()) / len(steered)
        hol_b = sum(v["holonomy"] for v in base.values()) / len(base)
        hol_s = sum(v["holonomy"] for v in steered.values()) / len(steered)
        print(f"  {'all':>5}  {rho_b:8.4f}  {rho_s:8.4f}  {rho_s-rho_b:+8.4f}  "
              f"{hol_b:9.4f}  {hol_s:9.4f}  {hol_s-hol_b:+8.4f}")

        if pointwise:
            rp = sum(d.relative_perturbation for d in pointwise.values()) / len(pointwise)
            print(f"  pointwise: mean relative perturbation {rp:.3f} "
                  f"across {len(pointwise)} steered layers")

        results["prompts"][prompt] = {
            "layers": prompt_rows,
            "mean": {"rho_baseline": rho_b, "rho_steered": rho_s,
                     "holonomy_baseline": hol_b, "holonomy_steered": hol_s},
        }

    print(f"\n{'='*74}")
    print("  INTERPRETATION")
    print(f"{'='*74}")
    all_means = [p["mean"] for p in results["prompts"].values()]
    if all_means:
        d_rho = sum(m["rho_steered"] - m["rho_baseline"] for m in all_means) / len(all_means)
        d_hol = sum(m["holonomy_steered"] - m["holonomy_baseline"] for m in all_means) / len(all_means)
        print(f"  Mean Δρ across prompts:       {d_rho:+.4f}")
        print(f"  Mean Δholonomy across prompts: {d_hol:+.4f} rad")
        if abs(d_rho) < 5e-3 and abs(d_hol) < 5e-3:
            print("  → Steering leaves context routing essentially unchanged:")
            print("    the intervention shifts representations pointwise without")
            print("    altering how attention composes context (a 'translation').")
        else:
            print("  → Steering measurably changes context routing: the intervention")
            print("    alters the order sensitivity / path dependence of attention,")
            print("    not just the position of representations.")

    if args.generate:
        prompt = prompts[0]
        print(f"\n{'='*74}")
        print("  SAMPLE GENERATIONS (first prompt)")
        print(f"{'='*74}")
        with llm.steering_disabled():
            base_text = llm.generate(prompt, max_new_tokens=60, do_sample=False)
        steered_text = llm.generate(prompt, max_new_tokens=60, do_sample=False)
        print(f"  baseline: {base_text[:300]}")
        print(f"  steered:  {steered_text[:300]}")
        results["generation"] = {"baseline": base_text, "steered": steered_text}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
