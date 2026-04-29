"""Tests for the embedding layer.

NOTE: First run downloads BAAI/bge-m3 (~570 MB). Subsequent runs use the
HuggingFace cache and are fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.retrieval.embeddings import (
    cosine_similarity_matrix,
    embed_column,
    embed_texts,
)


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    def test_shape(self):
        vecs = embed_texts(["hello world", "another sentence"])
        assert vecs.shape == (2, 1024)

    def test_dtype_float32(self):
        vecs = embed_texts(["test"])
        assert vecs.dtype == np.float32

    def test_normalised(self):
        vecs = embed_texts(["normalised vector test"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_list(self):
        vecs = embed_texts([])
        assert vecs.shape == (0, 1024)

    def test_deterministic(self):
        v1 = embed_texts(["product name column"])
        v2 = embed_texts(["product name column"])
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# embed_column / embed_target_field
# ---------------------------------------------------------------------------


class TestEmbedColumn:
    def test_shape(self):
        vec = embed_column("product_name", ["Dove Shampoo", "Luminos Wash"])
        assert vec.shape == (1024,)

    def test_samples_truncated_to_5(self):
        # Should not raise even with 20 samples
        vec = embed_column("x", [f"sample_{i}" for i in range(20)])
        assert vec.shape == (1024,)

    def test_no_samples(self):
        vec = embed_column("barcode", [])
        assert vec.shape == (1024,)


# ---------------------------------------------------------------------------
# Semantic similarity properties
# ---------------------------------------------------------------------------


class TestSemanticSimilarity:
    """Key property: semantically similar columns score higher than dissimilar ones."""

    @pytest.fixture(scope="class")
    def vecs(self):
        """Pre-compute all needed embeddings once for the whole class."""
        product_name_uk = embed_column("product_name", ["Dove Shampoo 250ml", "Luminos Body Wash"])
        prod_nm_in = embed_column("PROD_NM", ["Dove Shampoo 250ml", "Luminos Body Wash"])
        barcode = embed_column("barcode", ["5000112637922", "1819600133891"])
        sku_id = embed_column("sku_id", ["UK-00001", "UK-00002"])
        return {
            "product_name_uk": product_name_uk,
            "prod_nm_in": prod_nm_in,
            "barcode": barcode,
            "sku_id": sku_id,
        }

    def test_similar_columns_score_higher_than_dissimilar(self, vecs):
        """product_name (UK) vs PROD_NM (IN) should beat product_name vs barcode."""
        sim_similar = float(vecs["product_name_uk"] @ vecs["prod_nm_in"])
        sim_dissimilar = float(vecs["product_name_uk"] @ vecs["barcode"])
        assert sim_similar > sim_dissimilar, (
            f"Expected product_name~PROD_NM ({sim_similar:.3f}) > "
            f"product_name~barcode ({sim_dissimilar:.3f})"
        )

    def test_similar_columns_score_above_threshold(self, vecs):
        """Semantically equivalent columns (same samples, different name style) should be >0.85."""
        sim = float(vecs["product_name_uk"] @ vecs["prod_nm_in"])
        assert sim > 0.85, f"Expected >0.85, got {sim:.3f}"

    def test_dissimilar_columns_score_below_similar(self, vecs):
        """product_name vs sku_id should score lower than product_name vs PROD_NM."""
        sim_name_vs_name = float(vecs["product_name_uk"] @ vecs["prod_nm_in"])
        sim_name_vs_sku = float(vecs["product_name_uk"] @ vecs["sku_id"])
        assert sim_name_vs_name > sim_name_vs_sku


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_shape(self):
        a = embed_texts(["col_a_1", "col_a_2"])
        b = embed_texts(["col_b_1", "col_b_2", "col_b_3"])
        mat = cosine_similarity_matrix(a, b)
        assert mat.shape == (2, 3)

    def test_self_similarity_is_one(self):
        vecs = embed_texts(["product name"])
        mat = cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(mat[0, 0], 1.0, atol=1e-5)

    def test_values_in_range(self):
        a = embed_texts(["product name", "barcode number"])
        b = embed_texts(["sku identifier", "weight grams"])
        mat = cosine_similarity_matrix(a, b)
        assert mat.min() >= -1.01 and mat.max() <= 1.01
