#!/usr/bin/env python3
"""
Machine-POI: LLM Steering with Quran Text Embeddings

Demo script showing how to steer small LLMs using embeddings
derived from Quranic text.

Based on:
- Activation Addition (ActAdd): https://arxiv.org/abs/2308.10248
- Contrastive Activation Addition (CAA): https://arxiv.org/abs/2312.06681
- Eiffel Tower LLaMA: https://huggingface.co/spaces/dlouapre/eiffel-tower-llama
"""

import argparse
import sys
from pathlib import Path

from src import QuranSteerer
from config import (
    ExperimentConfig,
    LLM_MODELS,
    EMBEDDING_MODELS,
    STEERING_PRESETS,
    TEST_PROMPTS,
    get_recommended_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Steer LLMs using Quran text embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python main.py

  # Use specific model
  python main.py --llm qwen3-0.6b --embedding bge-m3

  # Adjust steering strength
  python main.py --preset strong --coefficient 0.8

  # Interactive mode
  python main.py --interactive

  # Run comparison on test prompts
  python main.py --compare
        """,
    )

    # Model selection
    parser.add_argument(
        "--llm",
        type=str,
        default="qwen2.5-0.5b",
        choices=list(LLM_MODELS.keys()) + ["custom"],
        help="LLM model to steer",
    )
    parser.add_argument(
        "--llm-path",
        type=str,
        default=None,
        help="Custom HuggingFace model path (when --llm=custom)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="bge-m3",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model for Quran text",
    )

    # Steering configuration
    parser.add_argument(
        "--preset",
        type=str,
        default="moderate",
        choices=list(STEERING_PRESETS.keys()),
        help="Steering intensity preset",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=None,
        help="Override steering coefficient (0.0-1.0)",
    )
    parser.add_argument(
        "--injection-mode",
        type=str,
        default=None,
        choices=["add", "blend", "replace", "clamp"],
        help="How to inject steering into activations (default: from preset)",
    )
    parser.add_argument(
        "--chunk-by",
        type=str,
        default="verse",
        choices=["verse", "paragraph", "surah"],
        help="How to chunk Quran text",
    )

    # Paths
    parser.add_argument(
        "--quran-path",
        type=str,
        default="al-quran.txt",
        help="Path to Quran text file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="vectors",
        help="Directory for cached embeddings/vectors",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device for computation",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization for LLM",
    )

    # Generation
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    # Mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive chat mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison on test prompts",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Theme for thematic steering (e.g., 'mercy', 'justice')",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize/Build the Knowledge Base index",
    )
    parser.add_argument(
        "--mra",
        action="store_true",
        help="Enable Multi-Resolution Analysis & Reasoning",
    )
    parser.add_argument(
        "--quran-persona",
        action="store_true",
        help="Enable Quran Persona steering (aggregates all resolutions)",
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                        Machine-POI                                ║
║         LLM Steering with Quran Text Embeddings                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Based on Activation Addition and Contrastive Activation Addition  ║
║  Paper: https://arxiv.org/abs/2308.10248                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def run_interactive(steerer: QuranSteerer, args):
    """Run interactive chat mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'compare' to toggle comparison mode")
    print("Type 'strength <value>' to adjust steering (0.0-1.0)")
    print()

    compare_mode = True

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if prompt.lower() == "compare":
            compare_mode = not compare_mode
            print(f"Comparison mode: {'ON' if compare_mode else 'OFF'}")
            continue

        if prompt.lower().startswith("strength "):
            try:
                value = float(prompt.split()[1])
                steerer.set_steering_strength(value)
                print(f"Steering strength set to {value}")
            except (ValueError, IndexError):
                print("Usage: strength <value>")
            continue

        if prompt.lower().startswith("theme "):
            theme = prompt.split(None, 1)[1] if len(prompt.split()) > 1 else None
            if theme:
                print(f"Switching to thematic steering: {theme}")
                steerer.prepare_thematic_steering(theme)
            continue

        # Generate response
        if compare_mode:
            print("\n--- Steered Output ---")
            steered = steerer.generate(prompt, max_new_tokens=args.max_tokens, mra_mode=args.mra)
            print(steered)

            print("\n--- Baseline Output ---")
            baseline = steerer.generate_unsteered(prompt, max_new_tokens=args.max_tokens)
            print(baseline)
        else:
            print("\n--- Output ---")
            output = steerer.generate(prompt, max_new_tokens=args.max_tokens, mra_mode=args.mra)
            print(output)


def run_comparison(steerer: QuranSteerer, args):
    """Run comparison on test prompts."""
    print("\n=== Running Comparison on Test Prompts ===\n")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {prompt}")
        print("-" * 60)

        steered, baseline = steerer.compare(prompt, max_new_tokens=args.max_tokens)

        print("STEERED:")
        print(steered[:300] + "..." if len(steered) > 300 else steered)
        print()
        print("BASELINE:")
        print(baseline[:300] + "..." if len(baseline) > 300 else baseline)
        print()
        print("=" * 60)
        print()


def run_single_prompt(steerer: QuranSteerer, prompt: str, args):
    """Run on a single prompt."""
    print(f"\nPrompt: {prompt}\n")
    print("-" * 60)

    steered, baseline = steerer.compare(prompt, max_new_tokens=args.max_tokens)

    print("STEERED OUTPUT:")
    print(steered)
    print()
    print("BASELINE OUTPUT:")
    print(baseline)


def main():
    args = parse_args()
    print_banner()

    # Create configuration
    config = get_recommended_config(
        llm_model=args.llm if args.llm != "custom" else args.llm_path,
        embedding_model=args.embedding,
        intensity=args.preset,
    )

    # Override with command line args
    if args.coefficient is not None:
        config.custom_coefficient = args.coefficient
    if args.device:
        config.device = args.device
    if args.quantize:
        config.quantization = args.quantize

    # Print configuration
    print(f"Configuration:")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Preset: {config.preset}")
    print(f"  Coefficient: {config.custom_coefficient or config.get_preset().coefficient}")
    print(f"  Injection Mode: {args.injection_mode or config.get_preset().injection_mode}")
    print(f"  Device: {config.device or 'auto'}")
    print(f"  Quantization: {config.quantization or 'none'}")
    print(f"  MRA Mode: {'ON' if args.mra else 'OFF'}")
    print()

    # Initialize steerer
    print("Initializing QuranSteerer...")
    steerer = QuranSteerer(
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        quran_path=args.quran_path,
        device=config.device,
        llm_quantization=config.quantization,
    )

    if args.init_db:
        steerer.initialize_knowledge_base()
        steerer.knowledge_base.build_index(args.quran_path)
        return

    # Load models
    print("Loading models (this may take a while)...")
    steerer.load_models()

    # Prepare steering
    print("Preparing Quran-based steering vectors...")
    cache_path = Path(args.cache_dir) / f"quran_{args.embedding}_{args.chunk_by}.npz"

    # Apply custom coefficient if specified
    if config.custom_coefficient:
        steerer.config.coefficient = config.custom_coefficient
    else:
        steerer.config.coefficient = config.get_preset().coefficient

    # Apply injection mode (CLI overrides preset)
    steerer.config.injection_mode = args.injection_mode or config.get_preset().injection_mode

    if args.theme:
        # First create base embeddings, then apply thematic
        steerer.prepare_quran_steering(
            chunk_by=args.chunk_by,
            cache_path=cache_path,
        )
        steerer.prepare_thematic_steering(args.theme)
    elif args.quran_persona:
        steerer.prepare_quran_persona(cache_dir=args.cache_dir)
    else:
        steerer.prepare_quran_steering(
            chunk_by=args.chunk_by,
            cache_path=cache_path,
        )

    print("Ready!\n")

    # Run appropriate mode
    if args.interactive:
        run_interactive(steerer, args)
    elif args.compare:
        run_comparison(steerer, args)
    elif args.prompt:
        # Note: run_single_prompt uses compare which calls compare_outputs which loops generate
        # We need to update run_single_prompt to pass mra_mode if we want it there
        # But for now, let's just make sure compare handles kwargs
        # steerer.compare calls llm.compare_outputs which calls generate(..., **kwargs)
        # So passing mra_mode=args.mra should work if we pass it to compare
        steered, baseline = steerer.compare(args.prompt, max_new_tokens=args.max_tokens, mra_mode=args.mra)
        print("STEERED OUTPUT:")
        print(steered)
        print()
        print("BASELINE OUTPUT:")
        print(baseline)
    else:
        # Default: run a demo prompt
        demo_prompt = "What is the meaning of life and how should we live?"
        # run_single_prompt(steerer, demo_prompt, args) 
        # Inline run_single_prompt logic to pass mra_mode easily:
        print(f"\nPrompt: {demo_prompt}\n")
        steered, baseline = steerer.compare(demo_prompt, max_new_tokens=args.max_tokens, mra_mode=args.mra)
        print("STEERED OUTPUT:")
        print(steered)

        print("\n" + "=" * 60)
        print("Try other modes:")
        print("  --interactive   Interactive chat mode")
        print("  --compare       Run on all test prompts")
        print("  --prompt 'X'    Test a specific prompt")
        print("  --theme mercy   Steer toward a specific theme")


if __name__ == "__main__":
    main()
