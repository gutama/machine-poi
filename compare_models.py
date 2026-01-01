#!/usr/bin/env python3
"""
Compare different models with Quran Steering.

This script allows you to run the same prompt across multiple models
to compare their responses and steering effectiveness.

Native Reasoning Modes:
- deepseek-r1-1.5b: Uses <think>...</think> blocks (temp=0.6, top_p=0.95)
- qwen3-0.6b: Native enable_thinking in chat template (temp=0.6, top_p=0.95)
- phi4-mini: Math reasoning focused (temp=0.8, top_p=0.95)

Models without native reasoning (generic fallback):
- qwen2.5-0.5b, smollm2-*, gemma-270m
"""
import argparse
import torch
import gc
import sys
from src import QuranSteerer
from config import LLM_MODELS

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models with Quran Steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default models
  ./compare_models.py
  
  # Compare reasoning models with native reasoning enabled
  ./compare_models.py --models deepseek-r1-1.5b qwen3-0.6b --reasoning
  
  # Compare with MRA mode
  ./compare_models.py --mra --prompt "How should I handle stress at work?"
  
  # List available models
  ./compare_models.py --list-models
        """
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["qwen2.5-0.5b", "qwen3-0.6b"], 
        help="List of models to compare (names from config.py)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="What is the meaning of life?", 
        help="Prompt to test"
    )
    parser.add_argument(
        "--reasoning", 
        action="store_true", 
        help="Enable native reasoning mode (uses model-specific config)"
    )
    parser.add_argument(
        "--mra", 
        action="store_true", 
        help="Enable Multi-Resolution Analysis (MRA) mode"
    )
    parser.add_argument(
        "--coefficient", 
        type=float, 
        default=0.5, 
        help="Steering coefficient"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models with their reasoning capabilities"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (reasoning mode may override)"
    )
    return parser.parse_args()


def list_models():
    """Print available models with reasoning info."""
    print("\nAvailable Models:")
    print("-" * 70)
    print(f"{'Model Name':<20} {'Reasoning Mode':<15} {'Description'}")
    print("-" * 70)
    
    reasoning_info = {
        "deepseek-r1-1.5b": ("deepseek", "Uses <think>...</think> blocks"),
        "phi4-mini": ("phi", "Math reasoning, no special tokens"),
        "qwen3-0.6b": ("qwen3", "Native enable_thinking in template"),
        "smollm3": (None, "No native reasoning"),
        "gemma-270m": (None, "No native reasoning"),
        "qwen2.5-0.5b": (None, "Standard instruct model"),
        "smollm2-135m": (None, "Compact model, no reasoning"),
        "smollm2-360m": (None, "Compact model, no reasoning"),
    }
    
    for model_name in LLM_MODELS.keys():
        mode, desc = reasoning_info.get(model_name, (None, "Unknown"))
        mode_str = mode if mode else "none"
        print(f"{model_name:<20} {mode_str:<15} {desc}")
    
    print("-" * 70)
    print("\nNative reasoning modes use model-specific parameters from documentation.")
    print("Use --reasoning to enable.\n")


def main():
    args = parse_args()
    
    if args.list_models:
        list_models()
        return
    
    results = {}
    
    print(f"\n{'='*60}")
    print("Machine-POI Model Comparison")
    print(f"{'='*60}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Prompt: {args.prompt}")
    print(f"Reasoning Mode: {args.reasoning} (uses native config if available)")
    print(f"MRA Mode: {args.mra}")
    print(f"Coefficient: {args.coefficient}")
    print(f"{'='*60}")
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model_name}")
        
        # Show reasoning config if applicable
        if args.reasoning and model_name in ["deepseek-r1-1.5b", "qwen3-0.6b", "phi4-mini"]:
            from src.llm_wrapper import SteeredLLM
            config = SteeredLLM.REASONING_CONFIGS.get(model_name, {})
            print(f"  Reasoning: {config.get('mode', 'generic')} mode")
            print(f"  Temperature: {config.get('temperature', 0.6)}")
            print(f"  Top-P: {config.get('top_p', 0.95)}")
        elif args.reasoning:
            print(f"  Reasoning: generic (step-by-step)")
        
        print(f"{'='*60}")
        
        steerer = None
        try:
            # Initialize steerer
            steerer = QuranSteerer(llm_model=model_name)
            
            # Load models
            steerer.load_models()
            
            # Configure steering
            steerer.config.coefficient = args.coefficient
            
            # Prepare steering vectors
            steerer.prepare_quran_steering()
            
            # Generate output
            output = steerer.generate(
                args.prompt, 
                use_domain_bridges=True,
                use_dynamic_steering=True,
                mra_mode=args.mra,
                reasoning_mode=args.reasoning,
                max_new_tokens=args.max_tokens
            )
            
            results[model_name] = output
            print(f"\nOutput ({model_name}):\n{output}\n")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results[model_name] = f"ERROR: {e}"
            
        finally:
            # Cleanup to free VRAM for next model
            if steerer:
                if steerer.llm:
                    del steerer.llm.model
                    del steerer.llm.tokenizer
                    del steerer.llm
                del steerer
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print Summary
    print("\n\n" + "="*80)
    print("COMPARISON RESULTS SUMMARY")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    if args.reasoning:
        print("Mode: Reasoning (native where available)")
    if args.mra:
        print("Mode: MRA (Multi-Resolution Analysis)")
    print()
    
    for model, output in results.items():
        print(f"--- Model: {model} ---")
        print(output.strip()[:500])  # Truncate for readability
        if len(output) > 500:
            print("... (truncated)")
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()
