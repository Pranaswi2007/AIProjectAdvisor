"""
pytest conftest — pre-warm the sentence-transformer embedding model
once for the entire test session so RAG integration tests don't each
pay the cold-start penalty (~15-30s on CPU).
"""
import pytest


def pytest_configure(config):
    """Load the embedding model into the global cache before any test runs."""
    try:
        from rag import get_model
        get_model()
    except Exception:
        pass  # if rag isn't importable (e.g. unit-only runs), that's fine
