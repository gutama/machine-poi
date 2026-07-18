#!/usr/bin/env python3
"""
Centered-Contrast Steering Probe

Companion to steered_vs_baseline_transport.py. That experiment steers with a
RAW mean activation of Quran verses; this probe measures how much of that
vector is Quran-specific at all, by comparing it against the mean activation
of neutral English sentences and steering with the centered contrast

    contrast_l = mean_l(quran verses) - mean_l(neutral sentences)

Reported per layer: |quran mean|, |neutral mean|, |contrast|, and
cos(quran, neutral). Then, for one prompt, pooled non-abelian ratio rho,
mean holonomy, mean relative perturbation, and a greedy generation under:

  A. baseline (no steering)
  B. raw mean-activation steering (the main experiment's vector)
  C. centered contrast steering at several coefficients

Usage:
    python experiments/centered_contrast_probe.py
    python experiments/centered_contrast_probe.py --model qwen3-0.6b --output probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_wrapper import SteeredLLM
from experiments.steered_vs_baseline_transport import (
    build_steering_vectors,
    force_eager_attention,
    load_quran_verses,
    select_workspace_layers,
    transport_summary,
)

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


def summarize(diag: dict) -> tuple:
    summary = transport_summary(diag)
    rho = sum(v["rho"] for v in summary.values()) / len(summary)
    hol = sum(v["holonomy"] for v in summary.values()) / len(summary)
    return rho, hol


def run_condition(llm: SteeredLLM, label: str, vectors, coefficient: float,
                  prompt: str, eta: float, max_loop_positions: int) -> dict:
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
    rho, hol = summarize(diag)
    pointwise = llm.get_steering_diagnostics()
    rel_pert = (
        sum(d.relative_perturbation for d in pointwise.values()) / len(pointwise)
        if pointwise else 0.0
    )
    text = llm.generate(prompt, max_new_tokens=50, do_sample=False)
    print(f"\n[{label}] coeff={coefficient}  rho={rho:.4f}  hol={hol:.4f}  "
          f"rel_pert={rel_pert:.3f}")
    print(f"  gen: {text[:220]!r}")
    return {"coefficient": coefficient, "rho": rho, "holonomy": hol,
            "relative_perturbation": rel_pert, "generation": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--model", default="smollm2-135m")
    parser.add_argument("--num-verses", type=int, default=12)
    parser.add_argument("--prompt",
                        default="What does it mean to live a just and merciful life?")
    parser.add_argument("--raw-coefficient", type=float, default=1.0)
    parser.add_argument("--centered-coefficients", type=float, nargs="*",
                        default=[1.0, 4.0, 8.0])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-loop-positions", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    quran_path = Path(__file__).parent.parent / "al-quran.txt"

    print(f"Loading model: {args.model}")
    llm = SteeredLLM(model_name=args.model)
    llm.load_model()
    force_eager_attention(llm)
    layers = select_workspace_layers(llm.num_layers)
    print(f"Workspace layers: {layers}")

    verses = load_quran_verses(quran_path, args.num_verses)
    quran_vecs = build_steering_vectors(llm, verses, layers)
    neutral_vecs = build_steering_vectors(llm, NEUTRAL_SENTENCES, layers)
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

    results = {"model": args.model, "prompt": args.prompt,
               "layers": layers, "geometry": geometry, "conditions": {}}
    conditions = results["conditions"]
    conditions["baseline"] = run_condition(
        llm, "A baseline", None, 0.0,
        args.prompt, args.eta, args.max_loop_positions)
    conditions["raw"] = run_condition(
        llm, f"B raw mean, c={args.raw_coefficient}", quran_vecs,
        args.raw_coefficient, args.prompt, args.eta, args.max_loop_positions)
    for c in args.centered_coefficients:
        conditions[f"centered_c{c}"] = run_condition(
            llm, f"C centered, c={c}", centered, c,
            args.prompt, args.eta, args.max_loop_positions)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
