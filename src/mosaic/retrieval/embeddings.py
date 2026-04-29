"""Embedding utilities for the ATLAS agent.

Uses BAAI/bge-m3 via sentence-transformers — multilingual, open-weights,
strong on noisy CPG column names and mixed-language samples.

The model is cached on first load (lazy singleton) so repeated calls within
a process pay the ~1s load cost only once.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "BAAI/bge-m3"
_EMBED_DIM = 1024
_MAX_SAMPLES = 5
_SAMPLE_MAX_CHARS = 50

# Lazy singleton — populated on first call to embed_texts()
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings with BGE-m3.

    Returns:
        np.ndarray of shape (n, 1024), float32.
    """
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


def embed_column(name: str, samples: list[str]) -> np.ndarray:
    """Build a descriptor for one source column and embed it.

    Descriptor format:
        "column_name: {name} | samples: {s1}, {s2}, ..."

    Uses up to 5 samples, each truncated to 50 chars.

    Returns:
        np.ndarray of shape (1024,), float32.
    """
    truncated = [str(s)[:_SAMPLE_MAX_CHARS] for s in samples[:_MAX_SAMPLES]]
    samples_str = ", ".join(truncated) if truncated else "(none)"
    descriptor = f"column_name: {name} | samples: {samples_str}"
    return embed_texts([descriptor])[0]


def embed_target_field(name: str, description: str, example_values: list[str]) -> np.ndarray:
    """Build a descriptor for one target schema field and embed it.

    Descriptor format:
        "field: {name} | description: {description} | examples: {e1}, {e2}, ..."

    Returns:
        np.ndarray of shape (1024,), float32.
    """
    truncated = [str(v)[:_SAMPLE_MAX_CHARS] for v in example_values[:_MAX_SAMPLES]]
    examples_str = ", ".join(truncated) if truncated else "(none)"
    descriptor = f"field: {name} | description: {description} | examples: {examples_str}"
    return embed_texts([descriptor])[0]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of a and every row of b.

    Both arrays must already be L2-normalised (BGE-m3 with normalize_embeddings=True).

    Returns:
        np.ndarray of shape (len(a), len(b)), values in [-1, 1].
    """
    return (a @ b.T).astype(np.float32)
