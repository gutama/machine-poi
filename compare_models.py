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

Embedding Models:
- bge-m3: BGE M3 (1024 dim, Arabic support)
- multilingual-e5: Multilingual E5 Large (1024 dim, Arabic support)
- qwen-embedding: Qwen2 7B Embedding (3584 dim, Arabic support)
- bge-large-zh: BGE Large Chinese (1024 dim, Chinese-focused)
- paraphrase-mpnet: Multilingual MPNet (768 dim, 50+ languages)
- paraphrase-minilm: Multilingual MiniLM (384 dim, 50+ languages, fast)
"""
import argparse
import torch
import gc
import sys
import time
from src import QuranSteerer
from config import LLM_MODELS, EMBEDDING_MODELS

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

  # Compare embedding models
  ./compare_models.py --compare-embeddings --embeddings bge-m3 multilingual-e5

  # List available models
  ./compare_models.py --list-models
        """
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["deepseek-r1-1.5b", "qwen3-0.6b"],
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
    # Embedding model comparison
    parser.add_argument(
        "--compare-embeddings",
        action="store_true",
        help="Compare embedding models instead of LLMs"
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["bge-m3", "multilingual-e5"],
        help="List of embedding models to compare"
    )
    parser.add_argument(
        "--embedding-queries",
        nargs="+",
        default=["mercy and compassion", "justice and fairness", "patience in hardship"],
        help="Queries to test embedding similarity"
    )
    return parser.parse_args()


def list_models():
    """Print available models with reasoning info."""
    print("\n" + "=" * 70)
    print("LLM MODELS")
    print("=" * 70)
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
    print("Use --reasoning to enable native reasoning modes.\n")

    # List embedding models
    print("=" * 70)
    print("EMBEDDING MODELS")
    print("=" * 70)
    print(f"{'Model Name':<20} {'Dim':<8} {'Arabic':<8} {'Memory':<10} {'HuggingFace Path'}")
    print("-" * 70)

    for name, info in EMBEDDING_MODELS.items():
        arabic = "Yes" if info.get("supports_arabic", False) else "No"
        mem = f"{info.get('memory_gb', 0):.1f} GB"
        print(f"{name:<20} {info['embedding_dim']:<8} {arabic:<8} {mem:<10} {info['hf_path']}")

    print("-" * 70)
    print("Use --compare-embeddings to compare embedding models.\n")


def compare_embeddings(args):
    """Compare embedding models on query-to-Quran retrieval."""
    from src.quran_embeddings import QuranEmbeddings
    from src.knowledge_base import QuranKnowledgeBase
    import numpy as np

    print(f"\n{'='*70}")
    print("EMBEDDING MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Models: {', '.join(args.embeddings)}")
    print(f"Test Queries: {args.embedding_queries}")
    print(f"{'='*70}\n")

    results = {}

    for embedding_name in args.embeddings:
        if embedding_name not in EMBEDDING_MODELS:
            print(f"Warning: Unknown embedding model '{embedding_name}', skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Testing Embedding: {embedding_name}")
        info = EMBEDDING_MODELS[embedding_name]
        print(f"  Dimension: {info['embedding_dim']}")
        print(f"  Arabic Support: {info.get('supports_arabic', False)}")
        print(f"  Memory: {info.get('memory_gb', 0):.1f} GB")
        print(f"{'='*60}")

        embedder = None
        kb = None
        try:
            # Initialize and load embedding model
            start_time = time.time()
            embedder = QuranEmbeddings(model_name=embedding_name)
            embedder.load_model()
            load_time = time.time() - start_time
            print(f"  Model load time: {load_time:.2f}s")

            # Load Quran text
            chunks = embedder.load_quran_text("al-quran.txt", chunking="verse")
            print(f"  Loaded {len(chunks)} Quran verses")

            # Create embeddings
            start_time = time.time()
            embeddings = embedder.create_embeddings(chunks[:100])  # First 100 for speed
            embed_time = time.time() - start_time
            print(f"  Embedding time (100 verses): {embed_time:.2f}s")
            print(f"  Embedding shape: {embeddings.shape}")

            # Test query similarity
            model_results = {
                "load_time": load_time,
                "embed_time": embed_time,
                "dim": embeddings.shape[1],
                "queries": {}
            }

            for query in args.embedding_queries:
                start_time = time.time()
                query_embedding = embedder.create_embeddings([query])[0]
                query_time = time.time() - start_time

                # Compute cosine similarities
                similarities = np.dot(embeddings, query_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                top_indices = np.argsort(similarities)[-3:][::-1]

                print(f"\n  Query: '{query}'")
                print(f"  Query embedding time: {query_time*1000:.1f}ms")
                print(f"  Top 3 matches:")
                for i, idx in enumerate(top_indices):
                    score = similarities[idx]
                    text = chunks[idx][:80] + "..." if len(chunks[idx]) > 80 else chunks[idx]
                    print(f"    {i+1}. [score={score:.4f}] {text}")

                model_results["queries"][query] = {
                    "query_time_ms": query_time * 1000,
                    "top_scores": [float(similarities[i]) for i in top_indices],
                    "top_texts": [chunks[i][:100] for i in top_indices]
                }

            results[embedding_name] = model_results

        except Exception as e:
            print(f"  Error: {e}")
            results[embedding_name] = {"error": str(e)}

        finally:
            # Cleanup
            if embedder and hasattr(embedder, 'model'):
                del embedder.model
            if embedder:
                del embedder
            if kb:
                del kb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print comparison summary
    print("\n\n" + "=" * 80)
    print("EMBEDDING COMPARISON SUMMARY")
    print("=" * 80)

    # Performance table
    print(f"\n{'Model':<20} {'Load (s)':<12} {'Embed (s)':<12} {'Dim':<8} {'Avg Score':<12}")
    print("-" * 70)
    for model_name, data in results.items():
        if "error" in data:
            print(f"{model_name:<20} ERROR: {data['error'][:40]}")
            continue
        avg_score = np.mean([
            np.mean(q["top_scores"])
            for q in data["queries"].values()
        ])
        print(f"{model_name:<20} {data['load_time']:<12.2f} {data['embed_time']:<12.2f} {data['dim']:<8} {avg_score:<12.4f}")

    print("-" * 70)
    print("\nHigher average scores indicate better semantic matching to Quran content.")

    return results


def main():
    args = parse_args()
    
    if args.list_models:
        list_models()
        return

    if args.compare_embeddings:
        compare_embeddings(args)
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
