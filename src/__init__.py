"""
Machine-POI: LLM Steering using Quran Text Embeddings.

This package implements activation steering for small language models using
text embeddings derived from Quranic verses.

Public classes are loaded lazily so lightweight imports such as
``src.workspace_diagnostics`` do not require optional runtime dependencies for
the full retrieval and model-loading stack.
"""

__version__ = "0.1.0"

_PUBLIC_IMPORTS = {
    "QuranEmbeddings": (".quran_embeddings", "QuranEmbeddings"),
    "SteeringVectorExtractor": (".steering_vectors", "SteeringVectorExtractor"),
    "ContrastiveSteeringExtractor": (
        ".steering_vectors",
        "ContrastiveSteeringExtractor",
    ),
    "SteeredLLM": (".llm_wrapper", "SteeredLLM"),
    "QuranSteerer": (".steerer", "QuranSteerer"),
    "ContrastiveQuranSteerer": (".steerer", "ContrastiveQuranSteerer"),
}

__all__ = list(_PUBLIC_IMPORTS)


def __getattr__(name):
    """Lazily import public classes on first attribute access."""
    if name not in _PUBLIC_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _PUBLIC_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
