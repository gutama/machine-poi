#!/usr/bin/env python3
"""
Experiment Reproduction Script for Machine-POI Paper

This script reproduces the experimental results described in PAPER.md:
1. Qualitative comparison (steered vs unsteered outputs)
2. Thematic consistency analysis
3. Coefficient sensitivity testing

Usage:
    python experiments/reproduce_paper.py [--model MODEL] [--quick]

Requirements:
    - GPU with at least 8GB VRAM (recommended)
    - Dependencies from requirements.txt installed
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steerer import QuranSteerer


def run_qualitative_comparison(steerer: QuranSteerer, prompts: list[str]) -> dict:
    """
    Run qualitative comparison between steered and unsteered outputs.
    Reproduces Section 5.1 of the paper.
    """
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON (Section 5.1)")
    print("=" * 70)
    
    results = []
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 50)
        
        # Generate unsteered
        unsteered = steerer.generate_unsteered(prompt, max_new_tokens=150)
        print(f"\n🔹 Unsteered:\n{unsteered[:500]}...")
        
        # Generate steered
        steered = steerer.generate(prompt, max_new_tokens=150)
        print(f"\n🔸 Steered (Quran Persona):\n{steered[:500]}...")
        
        results.append({
            "prompt": prompt,
            "unsteered": unsteered,
            "steered": steered,
        })
    
    return results


def run_mra_comparison(steerer: QuranSteerer, prompt: str) -> dict:
    """
    Run MRA (Multi-Resolution Analysis) mode comparison.
    Tests domain bridging functionality.
    """
    print("\n" + "=" * 70)
    print("MRA MODE WITH DOMAIN BRIDGING")
    print("=" * 70)
    
    print(f"\n📝 Prompt: {prompt}")
    print("-" * 50)
    
    # Generate with MRA mode
    mra_output = steerer.generate(
        prompt, 
        max_new_tokens=200, 
        mra_mode=True,
        use_domain_bridges=True
    )
    print(f"\n🔸 MRA Output:\n{mra_output}")
    
    return {"prompt": prompt, "mra_output": mra_output}


def run_thematic_analysis(steerer: QuranSteerer, theme: str, prompts: list[str]) -> dict:
    """
    Run thematic consistency analysis.
    Reproduces Section 5.2 of the paper.
    """
    print("\n" + "=" * 70)
    print(f"THEMATIC CONSISTENCY ANALYSIS: '{theme}' (Section 5.2)")
    print("=" * 70)
    
    theme_keywords = {
        "mercy": ["mercy", "merciful", "compassion", "kind", "forgive", "rahma"],
        "patience": ["patience", "patient", "sabr", "endure", "persevere"],
        "justice": ["justice", "just", "fair", "equality", "right"],
    }
    
    keywords = theme_keywords.get(theme.lower(), [theme.lower()])
    
    steered_keyword_count = 0
    unsteered_keyword_count = 0
    steered_religious_refs = 0
    unsteered_religious_refs = 0
    
    religious_markers = ["allah", "god", "divine", "quran", "prophet", "faith", "worship"]
    
    for prompt in prompts:
        steered, unsteered = steerer.compare(prompt, max_new_tokens=100)
        
        # Count keywords
        for kw in keywords:
            steered_keyword_count += steered.lower().count(kw)
            unsteered_keyword_count += unsteered.lower().count(kw)
        
        # Count religious references
        for marker in religious_markers:
            if marker in steered.lower():
                steered_religious_refs += 1
            if marker in unsteered.lower():
                unsteered_religious_refs += 1
    
    n = len(prompts)
    results = {
        "theme": theme,
        "prompts_tested": n,
        "steered_keyword_freq": steered_keyword_count / n,
        "unsteered_keyword_freq": unsteered_keyword_count / n,
        "steered_religious_rate": (steered_religious_refs / n) * 100,
        "unsteered_religious_rate": (unsteered_religious_refs / n) * 100,
    }
    
    print(f"\n📊 Results for theme '{theme}':")
    print(f"   Keyword frequency (steered):   {results['steered_keyword_freq']:.1f} / response")
    print(f"   Keyword frequency (unsteered): {results['unsteered_keyword_freq']:.1f} / response")
    print(f"   Religious refs (steered):      {results['steered_religious_rate']:.0f}%")
    print(f"   Religious refs (unsteered):    {results['unsteered_religious_rate']:.0f}%")
    
    return results


def run_coefficient_sensitivity(steerer: QuranSteerer, prompt: str, coefficients: list[float]) -> list[dict]:
    """
    Test sensitivity to steering coefficient.
    Reproduces Section 5.3 of the paper.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT SENSITIVITY ANALYSIS (Section 5.3)")
    print("=" * 70)
    
    print(f"\n📝 Test Prompt: {prompt}")
    
    results = []
    
    for coef in coefficients:
        steerer.set_steering_strength(coef)
        output = steerer.generate(prompt, max_new_tokens=100)
        
        result = {
            "coefficient": coef,
            "output_length": len(output),
            "output_preview": output[:200] + "..." if len(output) > 200 else output,
        }
        results.append(result)
        
        print(f"\n   α = {coef}:")
        print(f"   Output: {result['output_preview']}")
    
    # Reset to default
    steerer.set_steering_strength(0.5)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Reproduce Machine-POI paper experiments")
    parser.add_argument("--model", default="deepseek-r1-1.5b", help="LLM model to use")
    parser.add_argument("--embedding", default="paraphrase-minilm", help="Embedding model to use")
    parser.add_argument("--quick", action="store_true", help="Run quick version with fewer prompts")
    parser.add_argument("--section", choices=["all", "5.1", "5.2", "5.3", "mra"], 
                        default="all", help="Which section to reproduce")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MACHINE-POI PAPER REPRODUCTION")
    print(f"Model: {args.model} | Embedding: {args.embedding}")
    print("=" * 70)
    
    # Initialize steerer
    print("\n🔄 Loading models...")
    steerer = QuranSteerer(
        llm_model=args.model,
        embedding_model=args.embedding,
    )
    steerer.load_models()
    
    # Prepare Quran Persona steering
    print("\n🔄 Preparing Quran Persona steering vectors...")
    steerer.prepare_quran_persona(cache_dir="vectors")
    
    # Define test prompts
    qualitative_prompts = [
        "What is the meaning of life?",
        "How should I deal with a bug in my code?",
    ]
    
    mercy_prompts = [
        "Describe what mercy means.",
        "How should we show mercy to others?",
        "Tell me about compassion and kindness.",
    ]
    
    if args.quick:
        qualitative_prompts = qualitative_prompts[:1]
        mercy_prompts = mercy_prompts[:2]
    
    # Run experiments based on selection
    if args.section in ["all", "5.1"]:
        run_qualitative_comparison(steerer, qualitative_prompts)
    
    if args.section in ["all", "mra"]:
        run_mra_comparison(steerer, "How should I deal with a bug in my code?")
    
    if args.section in ["all", "5.2"]:
        run_thematic_analysis(steerer, "mercy", mercy_prompts)
    
    if args.section in ["all", "5.3"]:
        run_coefficient_sensitivity(
            steerer, 
            "What is patience?", 
            [0.2, 0.5, 0.8, 1.2]
        )
    
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT REPRODUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
