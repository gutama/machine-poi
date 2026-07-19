#!/usr/bin/env python3
"""
Centered-Contrast Steering Probe

Companion to steered_vs_baseline_transport.py. That experiment now steers
with the CAA-style centered contrast by default

    contrast_l = mean_l(quran verses) - mean_l(neutral sentences)

(--uncentered restores the legacy raw mean). This probe dissects that
vector: it separates the Quran-specific contrast direction from the generic
mean-activation component, reports their geometry, and compares steering
with each at controlled doses.

Reported per layer: |quran mean|, |neutral mean|, |contrast|, and
cos(quran, neutral). Then, for one prompt, per-layer and pooled non-abelian
ratio rho, mean holonomy, mean relative perturbation, and a greedy
generation under:

Note on prompts: transport diagnostics run on the RAW --prompt text (kept
comparable with steered_vs_baseline_transport.py), while generations use the
tokenizer's chat template when one exists so instruct baselines don't emit
EOS immediately. Both strings are recorded in the output JSON ("prompt" and
"generation_prompt") since rho/holonomy correspond to the former and the
generated text to the latter.

  A. baseline (steering vectors attached at coefficient 0 -- identical to no
     steering, but captures activation norms for coefficient calibration)
  B. optionally, raw mean-activation steering (--raw-coefficient)
  C. centered contrast steering at explicit coefficients and/or at a
     coefficient calibrated so every steered layer's relative perturbation
     stays at or below --target-perturbation

Usage:
    python experiments/centered_contrast_probe.py
    python experiments/centered_contrast_probe.py --model google/gemma-4-E2B-it \
        --dtype bfloat16 --layers 5 6 7 8 9 10 11 --target-perturbation 0.1 \
        --centered-coefficients --output probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_wrapper import SteeredLLM
from experiments.steered_vs_baseline_transport import (
    NEUTRAL_SENTENCES,
    force_eager_attention,
    load_quran_verses,
    mean_activation_vectors,
    select_workspace_layers,
    transport_summary,
)


def chat_prompt(llm: SteeredLLM, prompt: str) -> str:
    """Chat-templated prompt for instruct models; raw prompt otherwise."""
    tokenizer = llm.tokenizer
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def run_condition(llm: SteeredLLM, label: str, vectors, coefficient: float,
                  prompt: str, gen_prompt: str, eta: float,
                  max_loop_positions: int) -> dict:
    llm.clear_steering()
    if vectors is not None:
        for layer_idx, vector in vectors.items():
            llm.register_steering_hook(
                layer_idx, steering_vector=vector,
                coefficient=coefficient, injection_mode="add",
            )
    diag = llm.get_attention_transport_diagnostics(
        prompt, eta=eta, max_loop_positions=max_loop_positions
    )
    layers = {
        layer_idx: {"rho": summary["rho"], "holonomy": summary["holonomy"]}
        for layer_idx, summary in transport_summary(diag).items()
    }
    if layers:
        rho = sum(v["rho"] for v in layers.values()) / len(layers)
        hol = sum(v["holonomy"] for v in layers.values()) / len(layers)
    else:
        print("  [warn] no diagnostics returned (attention weights unavailable?)")
        rho = hol = float("nan")
    pointwise = llm.get_steering_diagnostics()
    per_layer_steering = {
        layer_idx: {"activation_norm": d.activation_norm,
                    "steering_norm": d.steering_norm,
                    "relative_perturbation": d.relative_perturbation}
        for layer_idx, d in pointwise.items()
    }
    rel_pert = (
        sum(d.relative_perturbation for d in pointwise.values()) / len(pointwise)
        if pointwise else 0.0
    )
    text = llm.generate(gen_prompt, max_new_tokens=60, do_sample=False)
    print(f"\n[{label}] coeff={coefficient:.4g}  rho={rho:.4f}  hol={hol:.4f}  "
          f"rel_pert={rel_pert:.3f}")
    print(f"  gen: {text[:220]!r}")
    return {"coefficient": coefficient, "rho": rho, "holonomy": hol,
            "relative_perturbation": rel_pert, "generation": text,
            "layers": layers, "steering": per_layer_steering}


def print_layer_deltas(baseline: dict, condition: dict, label: str,
                       steer_layers: list) -> None:
    print(f"\n  Per-layer deltas vs baseline [{label}]:")
    print(f"  {'layer':>5}  {'rho base':>9}  {'rho cond':>9}  {'d_rho':>8}  "
          f"{'hol base':>9}  {'hol cond':>9}  {'d_hol':>8}  steered?")
    for layer_idx in sorted(baseline["layers"], key=int):
        b = baseline["layers"][layer_idx]
        c = condition["layers"].get(layer_idx, b)
        mark = "<" if int(layer_idx) in steer_layers else ""
        print(f"  {int(layer_idx):5d}  {b['rho']:9.4f}  {c['rho']:9.4f}  "
              f"{c['rho'] - b['rho']:+8.4f}  {b['holonomy']:9.4f}  "
              f"{c['holonomy']:9.4f}  {c['holonomy'] - b['holonomy']:+8.4f}  {mark}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--model", default="smollm2-135m")
    parser.add_argument("--dtype", default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype override (e.g. bfloat16 to fit a "
                             "larger model in RAM on CPU)")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Steering layer indices (default: workspace band)")
    parser.add_argument("--num-verses", type=int, default=12)
    parser.add_argument("--prompt",
                        default="What does it mean to live a just and merciful life?")
    parser.add_argument("--raw-coefficient", type=float, default=None,
                        help="Also run raw mean-activation steering at this "
                             "coefficient (omit to skip)")
    parser.add_argument("--centered-coefficients", type=float, nargs="*",
                        default=[1.0, 4.0, 8.0])
    parser.add_argument("--target-perturbation", type=float, default=None,
                        help="Add a centered condition whose coefficient is "
                             "calibrated so every steered layer's relative "
                             "perturbation is at most this value")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-loop-positions", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    quran_path = Path(__file__).parent.parent / "al-quran.txt"

    print(f"Loading model: {args.model}")
    torch_dtype = getattr(torch, args.dtype) if args.dtype else None
    llm = SteeredLLM(model_name=args.model, torch_dtype=torch_dtype)
    llm.load_model()
    force_eager_attention(llm)
    layers = args.layers if args.layers else select_workspace_layers(llm.num_layers)
    print(f"Steering layers: {layers}")

    verses = load_quran_verses(quran_path, args.num_verses)
    quran_vecs = mean_activation_vectors(llm, verses, layers)
    neutral_vecs = mean_activation_vectors(llm, NEUTRAL_SENTENCES, layers)
    centered = {li: quran_vecs[li] - neutral_vecs[li] for li in layers}

    geometry = {}
    print("\nVector geometry per layer:")
    for li in layers:
        q, n, c = quran_vecs[li], neutral_vecs[li], centered[li]
        cos = torch.nn.functional.cosine_similarity(q, n, dim=0).item()
        print(f"  L{li}: |quran|={q.norm():.1f}  |neutral|={n.norm():.1f}  "
              f"|contrast|={c.norm():.1f}  cos(quran, neutral)={cos:.4f}")
        geometry[li] = {"quran_norm": q.norm().item(),
                        "neutral_norm": n.norm().item(),
                        "contrast_norm": c.norm().item(),
                        "cos_quran_neutral": cos}

    gen_prompt = chat_prompt(llm, args.prompt)
    common = dict(prompt=args.prompt, gen_prompt=gen_prompt, eta=args.eta,
                  max_loop_positions=args.max_loop_positions)

    results = {"model": args.model, "prompt": args.prompt,
               "generation_prompt": gen_prompt,
               "layers": layers, "geometry": geometry, "conditions": {}}
    conditions = results["conditions"]

    # Baseline with coefficient-0 hooks: identical forward pass to no
    # steering, but captures per-layer activation norms for calibration.
    baseline = run_condition(llm, "A baseline", centered, 0.0, **common)
    conditions["baseline"] = baseline

    if args.raw_coefficient is not None:
        conditions["raw"] = run_condition(
            llm, f"B raw mean, c={args.raw_coefficient}", quran_vecs,
            args.raw_coefficient, **common)

    for c in args.centered_coefficients:
        conditions[f"centered_c{c}"] = run_condition(
            llm, f"C centered, c={c}", centered, c, **common)

    if args.target_perturbation is not None:
        ratios = [
            d["activation_norm"] / d["steering_norm"]
            for d in baseline["steering"].values() if d["steering_norm"] > 0
        ]
        if not ratios:
            print("\n[warn] no calibration data; skipping target condition")
        else:
            c_target = args.target_perturbation * min(ratios)
            label = f"C centered, calibrated c={c_target:.3f} " \
                    f"(max rel_pert <= {args.target_perturbation})"
            condition = run_condition(llm, label, centered, c_target, **common)
            conditions["centered_target"] = condition
            print_layer_deltas(baseline, condition, label, layers)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
