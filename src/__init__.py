"""
Machine-POI: LLM Steering using Quran Text Embeddings

This package implements activation steering for small language models
using text embeddings derived from Quranic verses.
"""

from .quran_embeddings import QuranEmbeddings
from .steering_vectors import SteeringVectorExtractor, ContrastiveSteeringExtractor
from .llm_wrapper import SteeredLLM
from .steerer import QuranSteerer, ContrastiveQuranSteerer

__version__ = "0.1.0"
__all__ = [
    "QuranEmbeddings",
    "SteeringVectorExtractor",
    "ContrastiveSteeringExtractor",
    "SteeredLLM",
    "QuranSteerer",
    "ContrastiveQuranSteerer",
]
