#!/usr/bin/env python3
"""
Compare different models with Quran Steering.

This script allows you to run the same prompt across multiple models
to compare their responses and steering effectiveness.
"""
import argparse
import torch
import gc
import sys
from src import QuranSteerer
from config import LLM_MODELS

def parse_args():
    parser = argparse.ArgumentParser(description="Compare models with Quran Steering")
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["qwen2.5-0.5b", "deepseek-r1-1.5b"], 
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
        help="Enable reasoning mode (lower temp, step-by-step)"
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    results = {}
    
    print(f"Comparing models: {args.models}")
    print(f"Prompt: {args.prompt}")
    print(f"Reasoning Mode: {args.reasoning}")
    print(f"MRA Mode: {args.mra}")
    print("-" * 50)
    
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*50}")
        
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
                max_new_tokens=200
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
    print(f"Prompt: {args.prompt}\n")
    
    for model, output in results.items():
        print(f"--- Model: {model} ---")
        print(output.strip())
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()
