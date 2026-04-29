"""Unit tests for hyperplane enumeration."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt.hyperplanes import (
    build_transform_matrix,
    generate_planes_to_index,
    lift,
    symmetrize,
)


def test_generate_planes_dimension_2_max_weight_1() -> None:
    rows = generate_planes_to_index(2, max_weight=1)
    arr = np.asarray(rows)
    # All rows should be lex-sorted, first column == 1, values in [0, 1]
    assert (arr[:, 0] == 1.0).all()
    assert (arr >= 0.0).all() and (arr <= 1.0).all()
    # Pin the actual enumeration: dim=2, max_weight=1, normalized to
    # leading 1 → only (1, 0) and (1, 1) survive after dedup.
    np.testing.assert_array_equal(arr, np.array([[1.0, 0.0], [1.0, 1.0]]))


def test_generate_planes_max_weight_zero_returns_identity() -> None:
    rows = generate_planes_to_index(3, max_weight=0)
    arr = np.asarray(rows)
    np.testing.assert_array_equal(arr, np.eye(3))


def test_symmetrize_grows_then_dedups() -> None:
    LCs = jnp.array([[1.0, 2.0]])
    out = symmetrize(LCs)
    # ±-parity gives 4 sign-variants, permutations multiply by 2 → 8.
    # Pruning removes (0,*) and (1,0)-style invalid forms, dedup
    # produces the 4 canonical normalised forms.
    assert out.shape[1] == 2
    assert out.shape[0] >= 2
    # Every row must start with 1 (after normalisation).
    np.testing.assert_allclose(np.asarray(out)[:, 0], 1.0, atol=1e-6)


def test_build_transform_matrix_includes_identity(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(50, 3)))
    A = build_transform_matrix(X, num_terms=2, max_weight=1, do_scaling=False)
    # First n_in columns of A.T should be identity (i.e., first n_in rows of A are I).
    n_in = 3
    np.testing.assert_allclose(np.asarray(A[:, :n_in]), np.eye(n_in), atol=1e-6)


def test_build_transform_matrix_lift_shape(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(20, 4)))
    A = build_transform_matrix(X, num_terms=2, max_weight=1)
    Xp = lift(X, A)
    assert Xp.shape[0] == 20
    assert Xp.shape[1] >= 4  # at least the identity block


def test_lift_preserves_original_features(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(10, 3)))
    A = build_transform_matrix(X, num_terms=2, max_weight=1, do_scaling=False)
    Xp = lift(X, A)
    np.testing.assert_allclose(np.asarray(Xp[:, :3]), np.asarray(X), atol=1e-12)


def test_dedup_warning_fires_when_aggressive_rounding(rng) -> None:
    """G-12: warn when aggressive rounding collapses many rows."""
    X = jnp.asarray(rng.uniform(-1.0, 1.0, size=(40, 4)))
    # tol_decimals=1 + max_weight=3 + num_terms=3 produces many near-duplicate
    # rows after rounding; expect the warning to fire.
    with pytest.warns(UserWarning, match=r"rounded.*dedup"):
        build_transform_matrix(
            X, num_terms=3, max_weight=3, tol_decimals=1, do_scaling=False
        )


def test_categorical_features_excluded(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(15, 4)))
    A_no_cat = build_transform_matrix(X, num_terms=2, max_weight=1)
    A_with_cat = build_transform_matrix(X, num_terms=2, max_weight=1, categorical_features=(0,))
    # Excluding feature 0 from permutations should give fewer (or equal) lifted columns.
    assert A_with_cat.shape[1] <= A_no_cat.shape[1]
