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
import asyncio
from pathlib import Path

from .knowledge_base import StaleIndexError
from .steerer import InvalidConfigError, QuranSteerer, SteeringConfig
from .config import (
    ExperimentConfig,
    LLM_MODELS,
    EMBEDDING_MODELS,
    STEERING_PRESETS,
    TEST_PROMPTS,
    get_recommended_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="machine-poi",
        description="Steer LLMs using Quran text embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python main.py

  # Use specific model
  python main.py --llm qwen3-0.6b --embedding bge-m3

  # Adjust the dose (target relative perturbation per layer)
  python main.py --preset strong --dose-ratio 0.08

  # Interactive mode
  python main.py --interactive

  # Run comparison on test prompts
  python main.py --compare
        """,
    )

    parser.add_argument("--guidance-config", help="Validated Quran-guidance JSON configuration")
    parser.add_argument("--guidance-mode", choices=("validate", "mock", "model"), default="validate")
    parser.add_argument("--guidance-output", help="Write the resolved configuration and evaluation report")

    # Model selection
    parser.add_argument(
        "--llm",
        type=str,
        default="deepseek-r1-1.5b",
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
        default="paraphrase-minilm",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model for Quran text",
    )

    parser.add_argument("--revision", help="Pinned Hugging Face LLM commit revision")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Opt into reviewed remote code; requires a full --revision commit")

    # Steering configuration
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(STEERING_PRESETS.keys()),
        help="Steering preset (default: 'moderate' with the model's recommended "
             "layers)",
    )
    dose = parser.add_mutually_exclusive_group()
    dose.add_argument(
        "--dose-ratio",
        type=float,
        default=None,
        help="Target relative perturbation per steered layer, calibrated from "
             "activation norms (-1 to 1; negative steers away; add mode only)",
    )
    dose.add_argument(
        "--coefficient",
        type=float,
        default=None,
        help="Raw steering coefficient instead of a dose ratio (0.0-2.0; blend mode "
             "0.0-1.0); required for blend, replace and clamp modes",
    )
    parser.add_argument(
        "--injection-mode",
        type=str,
        default=None,
        choices=["add", "blend", "replace", "clamp"],
        help="How to inject steering into activations (default: from preset)",
    )
    parser.add_argument(
        "--layer-distribution",
        type=str,
        default=None,
        choices=["uniform", "bell", "focused", "workspace"],
        help="How to distribute steering across layers (default: from preset)",
    )
    parser.add_argument(
        "--chunk-by",
        type=str,
        default=None,
        choices=["verse", "paragraph", "surah"],
        help="How to chunk Quran text (default: from preset)",
    )
    parser.add_argument(
        "--recipe",
        default="centered",
        choices=["centered", "raw_mean"],
        help="Vector recipe: Quran mean minus a neutral Arabic control mean "
             "(default), or the older uncentered mean",
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
        help="Sampling temperature (ignored with --greedy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed shared by both comparison arms (default: 42)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Decode greedily instead of sampling",
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
        "--rebuild",
        action="store_true",
        help="With --init-db, rebuild the vector index even if one exists "
             "(required after changing --embedding or the corpus)",
    )
    parser.add_argument(
        "--mra",
        action="store_true",
        help="Enable Multi-Resolution Analysis & Reasoning",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning mode (lower temp, step-by-step prompting)",
    )
    parser.add_argument(
        "--quran-persona",
        action="store_true",
        help="Enable Quran Persona steering (aggregates all resolutions)",
    )

    # LightRAG Knowledge Graph options
    parser.add_argument(
        "--graph-kb",
        action="store_true",
        help="Enable LightRAG knowledge graph for enhanced retrieval",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "gemini", "ollama"],
        help="LLM provider for graph entity extraction (default: openai)",
    )
    parser.add_argument(
        "--llm-api-model",
        type=str,
        default=None,
        help="Model name for LLM API (e.g., gpt-5.2, gemini-3.0-pro, qwen2.5:7b)",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build LightRAG graph index (use with --init-db)",
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                           Machine-POI                                 ║
║          LLM Steering with Quranic Semantic Embeddings                ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Features:                                                            ║
║    • Multi-Resolution Analysis (Verse/Passage/Surah)                  ║
║    • LightRAG Knowledge Graph (--graph-kb)                            ║
║    • Domain Bridging for Cross-Domain Analogies                       ║
║    • Quran Persona Steering                                           ║
║    • Contrastive Activation Addition (CAA)                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Based on: https://arxiv.org/abs/2308.10248 (ActAdd)                  ║
║            https://arxiv.org/abs/2312.06681 (CAA)                     ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)


def resolve_steering(args, config: ExperimentConfig):
    """Resolve steering settings: CLI flag > --preset > model recommendation.

    Returns the validated SteeringConfig and the text chunking to use.
    """
    preset = config.get_preset()
    injection_mode = args.injection_mode or preset.injection_mode
    if args.coefficient is not None:
        dose = {"dose_ratio": None, "coefficient": args.coefficient}
    elif injection_mode != "add":
        raise InvalidConfigError(
            f"--injection-mode {injection_mode} needs --coefficient; dose ratios "
            "apply to add mode"
        )
    else:
        ratio = args.dose_ratio if args.dose_ratio is not None else preset.dose_ratio
        dose = {"dose_ratio": ratio}
    if args.layer_distribution is not None:
        target_layers = None  # The explicit distribution selects layers.
    elif config.custom_layers is not None:
        target_layers = list(config.custom_layers)
    else:
        target_layers = preset.target_layers
    steering = SteeringConfig(
        **dose,
        target_layers=target_layers,
        injection_mode=injection_mode,
        layer_distribution=args.layer_distribution or preset.layer_distribution,
    )
    steering.validate()
    return steering, args.chunk_by or preset.chunk_by


async def build_graph_index(steerer: QuranSteerer, quran_path: str) -> None:
    """Initialize and build the graph index on one event loop.

    LightRAG storages bind to the loop that initializes them, so separate
    asyncio.run calls would build on objects tied to a closed loop.
    """
    await steerer.initialize_hybrid_knowledge_base()
    try:
        await steerer.hybrid_kb.build_index(quran_path, build_graph=True)
    finally:
        await steerer.hybrid_kb.finalize()


def generation_options(args) -> dict:
    """Generation settings shared by every CLI mode."""
    return {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "mra_mode": args.mra,
        "reasoning_mode": args.reasoning,
        "seed": args.seed,
        "do_sample": not args.greedy,
    }


def describe_dose(steering) -> str:
    """Describe the dose: a calibrated ratio, or the raw coefficient."""
    ratio = steering.get("dose_ratio")
    if ratio is not None:
        return f"dose ratio {ratio}"
    return f"coefficient {steering.get('coefficient')}"


def print_settings(steerer: QuranSteerer) -> None:
    """Print the decoding and steering settings the last run used."""
    settings = steerer.last_run_settings
    if not isinstance(settings, dict) or not settings:
        return
    decoding = (
        f"temperature {settings.get('temperature')}"
        if settings.get("do_sample")
        else "greedy"
    )
    steering = settings.get("steering", {})
    print(
        f"[seed {settings.get('seed')}, {decoding}, retrieval {settings.get('retrieval')}, "
        f"{describe_dose(steering)}, mode {steering.get('injection_mode')}, "
        f"chat template {settings.get('chat_template')}]"
    )


def run_interactive(steerer: QuranSteerer, args):
    """Run interactive chat mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'compare' to toggle comparison mode")
    print("Type 'strength <value>' to adjust the dose (a ratio in -1 to 1, or the raw "
          "coefficient when running with --coefficient)")
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
                if steerer.config.dose_ratio is None:
                    steerer.set_steering_strength(value)
                else:
                    steerer.set_dose_ratio(value)
                print(f"Steering set to {describe_dose(vars(steerer.config))}")
            except InvalidConfigError as exc:
                print(f"Strength unchanged: {exc}")
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
            steered, baseline = steerer.compare(prompt, **generation_options(args))
            print("\n--- Steered Output ---")
            print(steered)
            print("\n--- Baseline Output ---")
            print(baseline)
            print_settings(steerer)
        else:
            print("\n--- Output ---")
            print(steerer.generate(prompt, **generation_options(args)))


def run_comparison(steerer: QuranSteerer, args):
    """Run comparison on test prompts."""
    print("\n=== Running Comparison on Test Prompts ===\n")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {prompt}")
        print("-" * 60)

        steered, baseline = steerer.compare(prompt, **generation_options(args))

        print("STEERED:")
        print(steered[:300] + "..." if len(steered) > 300 else steered)
        print()
        print("BASELINE:")
        print(baseline[:300] + "..." if len(baseline) > 300 else baseline)
        print_settings(steerer)
        print()
        print("=" * 60)
        print()


def run_single_prompt(steerer: QuranSteerer, prompt: str, args):
    """Run on a single prompt."""
    print(f"\nPrompt: {prompt}\n")
    print("-" * 60)

    steered, baseline = steerer.compare(prompt, **generation_options(args))

    print("STEERED OUTPUT:")
    print(steered)
    print()
    print("BASELINE OUTPUT:")
    print(baseline)
    print_settings(steerer)


def main():
    try:
        run(parse_args())
    except StaleIndexError as exc:
        raise SystemExit(f"Stale vector index: {exc}")


def run(args):
    if getattr(args, "guidance_config", None):
        from .guidance_cli import run_guidance
        result = run_guidance(args.guidance_config, args.guidance_mode, args.guidance_output)
        print(f"{result['kind']}: {len(result['rows'])} trajectories")
        if result["status"] == "model_unavailable":
            raise SystemExit(1)
        return
    if args.llm == "custom" and not args.llm_path:
        raise SystemExit("--llm custom requires --llm-path")
    print_banner()

    # Create configuration
    config = get_recommended_config(
        llm_model=args.llm if args.llm != "custom" else args.llm_path,
        embedding_model=args.embedding,
        intensity=args.preset,
    )
    if args.device:
        config.device = args.device
    if args.quantize:
        config.quantization = args.quantize
    # Validate before loading any model.
    try:
        steering, chunk_by = resolve_steering(args, config)
    except InvalidConfigError as exc:
        raise SystemExit(f"Invalid steering configuration: {exc}")

    # Print configuration
    print("Configuration:")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Preset: {args.preset or f'model default ({config.preset} fallback)'}")
    print(f"  Dose: {describe_dose(vars(steering))}")
    print(f"  Injection Mode: {steering.injection_mode}")
    print(f"  Layer Distribution: {steering.layer_distribution}")
    print(f"  Target Layers: {steering.target_layers or 'from distribution'}")
    print(f"  Chunk By: {chunk_by}")
    print(f"  Device: {config.device or 'auto'}")
    print(f"  Quantization: {config.quantization or 'none'}")
    print(f"  MRA Mode: {'ON' if args.mra else 'OFF'}")
    print(f"  Graph KB: {'ON' if args.graph_kb else 'OFF'}")
    if args.graph_kb:
        print(f"  LLM Provider: {args.llm_provider}")
        print(f"  API Model: {args.llm_api_model or 'default'}")
    print()

    # Create LLM function for graph KB entity extraction
    llm_func = None
    if args.graph_kb:
        from .llm_adapters import (
            create_openai_adapter,
            create_gemini_adapter,
            create_ollama_adapter,
        )
        if args.llm_provider == "openai":
            model = args.llm_api_model or "gpt-4o-mini"
            llm_func = create_openai_adapter(model_name=model)
        elif args.llm_provider == "gemini":
            model = args.llm_api_model or "gemini-2.0-flash"
            llm_func = create_gemini_adapter(model_name=model)
        elif args.llm_provider == "ollama":
            model = args.llm_api_model or "qwen2.5:7b"
            llm_func = create_ollama_adapter(model_name=model)

    # Initialize steerer
    print("Initializing QuranSteerer...")
    steerer = QuranSteerer(
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        quran_path=args.quran_path,
        device=config.device,
        llm_quantization=config.quantization,
        use_graph_kb=args.graph_kb,
        llm_func=llm_func,
        llm_revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    steerer.config = steering

    if args.init_db:
        steerer.initialize_knowledge_base()
        steerer.knowledge_base.build_index(args.quran_path, rebuild=args.rebuild)
        if args.build_graph and args.graph_kb:
            print("Building LightRAG graph index (this may take a while)...")
            asyncio.run(build_graph_index(steerer, args.quran_path))
            print("Graph index built successfully!")
        return

    # Load models
    print("Loading models (this may take a while)...")
    steerer.load_models()

    # Prepare steering
    print("Preparing Quran-based steering vectors...")
    cache_path = Path(args.cache_dir) / f"quran_{args.embedding}_{chunk_by}_{args.recipe}.npz"

    if args.theme:
        steerer.prepare_thematic_steering(args.theme)
    elif args.quran_persona:
        steerer.prepare_quran_persona(cache_dir=args.cache_dir, recipe=args.recipe)
    else:
        steerer.prepare_quran_steering(
            chunk_by=chunk_by,
            cache_path=cache_path,
            recipe=args.recipe,
        )

    print("Ready!\n")

    # Run appropriate mode
    if args.interactive:
        run_interactive(steerer, args)
    elif args.compare:
        run_comparison(steerer, args)
    elif args.prompt:
        run_single_prompt(steerer, args.prompt, args)
    else:
        run_single_prompt(
            steerer, "What is the meaning of life and how should we live?", args
        )

        print("\n" + "=" * 60)
        print("Try other modes:")
        print("  --interactive   Interactive chat mode")
        print("  --compare       Run on all test prompts")
        print("  --prompt 'X'    Test a specific prompt")
        print("  --theme mercy   Steer toward a specific theme")


if __name__ == "__main__":
    main()
