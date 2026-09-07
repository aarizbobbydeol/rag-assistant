"""Tests for the dense vector stores.

`NumpyVectorStore` is the default and is covered exhaustively. The faiss, pgvector
and qdrant backends need a driver (and, for the last two, a live server) that CI
does not have, so they are covered at the seam that CI *can* reach: a missing
driver must surface as `ConfigurationError`, never a bare `ImportError`, and a
configured table name that is not a plain identifier must be rejected outright.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.errors import ConfigurationError
from app.retrieval.vectorstore import (
    NumpyVectorStore,
    PgVectorStore,
    QdrantStore,
    VectorStore,
    get_vector_store,
)

DIM = 8


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def basis(index: int, dim: int = DIM, scale: float = 1.0) -> np.ndarray:
    """A one-hot vector: basis vectors are mutually orthogonal, so cosines are exact."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = scale
    return vector


def matrix(*indices: int) -> np.ndarray:
    return np.vstack([basis(i) for i in indices]).astype(np.float32)


@pytest.fixture
def store() -> NumpyVectorStore:
    return NumpyVectorStore(DIM)


# --------------------------------------------------------------------------- #
# add / len / replace
# --------------------------------------------------------------------------- #
def test_add_grows_the_store(store: NumpyVectorStore) -> None:
    assert len(store) == 0
    store.add(["a", "b", "c"], matrix(0, 1, 2))
    assert len(store) == 3
    assert store.ids == ["a", "b", "c"]


def test_add_empty_batch_is_a_noop(store: NumpyVectorStore) -> None:
    store.add([], np.zeros((0, DIM), dtype=np.float32))
    assert len(store) == 0
    assert store.search(basis(0), 5) == []


def test_add_replaces_an_existing_id_in_place(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    store.add(["a"], basis(2).reshape(1, DIM))

    assert len(store) == 2, "replacement must not append a second row"
    assert store.ids == ["a", "b"]
    assert store.search(basis(2), 1) == [("a", pytest.approx(1.0))]
    # The old direction is gone, not merely outranked.
    assert store.search(basis(0), 2)[0][1] == pytest.approx(0.0)


def test_add_with_duplicate_ids_in_one_batch_keeps_the_last(store: NumpyVectorStore) -> None:
    store.add(["a", "a"], matrix(0, 3))
    assert len(store) == 1
    assert store.search(basis(3), 1)[0] == ("a", pytest.approx(1.0))


def test_add_normalises_non_unit_vectors(store: NumpyVectorStore) -> None:
    store.add(["a"], basis(0, scale=17.0).reshape(1, DIM))
    assert store.search(basis(0, scale=0.001), 1)[0][1] == pytest.approx(1.0)


def test_add_accepts_a_single_flat_vector(store: NumpyVectorStore) -> None:
    store.add(["a"], basis(0))
    assert len(store) == 1


def test_add_rejects_a_dimension_mismatch(store: NumpyVectorStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        store.add(["a"], np.ones((1, DIM + 1), dtype=np.float32))


def test_add_rejects_an_id_count_mismatch(store: NumpyVectorStore) -> None:
    with pytest.raises(ValueError, match="ids"):
        store.add(["a", "b"], matrix(0))


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_search_on_an_empty_store_returns_empty(store: NumpyVectorStore) -> None:
    assert store.search(basis(0), 5) == []


def test_search_returns_cosine_descending(store: NumpyVectorStore) -> None:
    near = np.zeros(DIM, dtype=np.float32)
    near[0] = 1.0
    near[1] = 1.0  # 45 degrees from basis(0)
    store.add(["far", "near", "exact"], np.vstack([basis(1), near, basis(0)]))

    hits = store.search(basis(0), 3)
    assert [chunk_id for chunk_id, _ in hits] == ["exact", "near", "far"]
    assert [score for _, score in hits] == sorted((s for _, s in hits), reverse=True)
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(1 / math.sqrt(2), abs=1e-6)
    assert hits[2][1] == pytest.approx(0.0, abs=1e-6)


def test_search_returns_negative_cosine_for_opposed_vectors(store: NumpyVectorStore) -> None:
    store.add(["opposite"], (-basis(0)).reshape(1, DIM))
    assert store.search(basis(0), 1)[0][1] == pytest.approx(-1.0)


def test_search_clamps_k_to_the_store_size(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    assert len(store.search(basis(0), 50)) == 2


def test_search_with_non_positive_k_returns_empty(store: NumpyVectorStore) -> None:
    store.add(["a"], matrix(0))
    assert store.search(basis(0), 0) == []
    assert store.search(basis(0), -3) == []


def test_search_rejects_a_query_of_the_wrong_dimension(store: NumpyVectorStore) -> None:
    store.add(["a"], matrix(0))
    with pytest.raises(ValueError, match="dimension"):
        store.search(np.ones(DIM + 2, dtype=np.float32), 1)


def test_search_breaks_ties_by_insertion_order(store: NumpyVectorStore) -> None:
    store.add(["first", "second", "third"], np.vstack([basis(0)] * 3))
    assert [cid for cid, _ in store.search(basis(0), 3)] == ["first", "second", "third"]


def test_search_is_repeatable(store: NumpyVectorStore) -> None:
    store.add(["a", "b", "c"], matrix(0, 1, 2))
    query = np.array([0.9, 0.4, 0.1, 0, 0, 0, 0, 0], dtype=np.float32)
    assert store.search(query, 3) == store.search(query, 3)


# --------------------------------------------------------------------------- #
# delete / clear
# --------------------------------------------------------------------------- #
def test_delete_removes_and_reports_the_count(store: NumpyVectorStore) -> None:
    store.add(["a", "b", "c"], matrix(0, 1, 2))
    assert store.delete(["b"]) == 1
    assert len(store) == 2
    assert store.ids == ["a", "c"]


def test_delete_ignores_unknown_and_duplicated_ids(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    assert store.delete(["zzz"]) == 0
    assert store.delete(["a", "a", "nope"]) == 1
    assert len(store) == 1


def test_delete_compacts_so_later_searches_stay_correct(store: NumpyVectorStore) -> None:
    store.add(["a", "b", "c", "d"], matrix(0, 1, 2, 3))
    store.delete(["a", "c"])

    assert store.ids == ["b", "d"]
    # Every surviving id must still map to its own vector, not to a neighbour's.
    assert store.search(basis(1), 2)[0] == ("b", pytest.approx(1.0))
    assert store.search(basis(3), 2)[0] == ("d", pytest.approx(1.0))
    # And the deleted directions must be unreachable.
    assert all(score == pytest.approx(0.0) for _, score in store.search(basis(0), 2))


def test_delete_then_readd_reuses_the_id_cleanly(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    store.delete(["a"])
    store.add(["a"], matrix(4))

    assert len(store) == 2
    assert store.ids == ["b", "a"]
    assert store.search(basis(4), 1)[0] == ("a", pytest.approx(1.0))


def test_delete_everything_leaves_a_usable_store(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    assert store.delete(["a", "b"]) == 2
    assert len(store) == 0
    assert store.search(basis(0), 3) == []
    store.add(["c"], matrix(2))
    assert store.search(basis(2), 1)[0][0] == "c"


def test_clear_empties_the_store(store: NumpyVectorStore) -> None:
    store.add(["a", "b"], matrix(0, 1))
    store.clear()
    assert len(store) == 0
    assert store.ids == []
    assert store.search(basis(0), 3) == []
    store.add(["a"], matrix(1))
    assert len(store) == 1


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_exactly(tmp_path: Path, store: NumpyVectorStore) -> None:
    query = np.array([0.7, 0.2, 0.5, 0, 0, 0, 0, 0.1], dtype=np.float32)
    store.add(["a", "b", "c"], matrix(0, 1, 2))
    store.delete(["b"])
    before = store.search(query, 5)

    target = tmp_path / "vectors"
    store.save(target)
    assert (tmp_path / "vectors.npz").exists()
    assert (tmp_path / "vectors.ids.json").exists()

    restored = NumpyVectorStore(DIM)
    restored.load(target)
    assert len(restored) == len(store)
    assert restored.ids == store.ids
    assert restored.search(query, 5) == before
    np.testing.assert_array_equal(restored.vectors(), store.vectors())


def test_save_load_round_trips_an_empty_store(tmp_path: Path, store: NumpyVectorStore) -> None:
    store.save(tmp_path / "empty")
    restored = NumpyVectorStore(DIM)
    restored.load(tmp_path / "empty")
    assert len(restored) == 0
    assert restored.search(basis(0), 3) == []


def test_load_replaces_rather_than_merges(tmp_path: Path, store: NumpyVectorStore) -> None:
    store.add(["a"], matrix(0))
    store.save(tmp_path / "vectors")

    other = NumpyVectorStore(DIM)
    other.add(["stale"], matrix(5))
    other.load(tmp_path / "vectors")
    assert other.ids == ["a"]


def test_save_creates_missing_directories(tmp_path: Path, store: NumpyVectorStore) -> None:
    store.add(["a"], matrix(0))
    store.save(tmp_path / "nested" / "dir" / "vectors")
    restored = NumpyVectorStore(DIM)
    restored.load(tmp_path / "nested" / "dir" / "vectors")
    assert restored.ids == ["a"]


def test_load_missing_file_raises(tmp_path: Path, store: NumpyVectorStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "does-not-exist")


def test_load_rejects_a_different_dimension(tmp_path: Path, store: NumpyVectorStore) -> None:
    store.add(["a"], matrix(0))
    store.save(tmp_path / "vectors")
    with pytest.raises(ValueError, match="dim"):
        NumpyVectorStore(DIM + 1).load(tmp_path / "vectors")


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def test_get_vector_store_returns_a_working_store(settings: Settings) -> None:
    store = get_vector_store(settings, DIM)
    assert isinstance(store, VectorStore)
    assert store.dim == DIM

    store.add(["a", "b"], matrix(0, 1))
    assert len(store) == 2
    assert store.search(basis(1), 1)[0][0] == "b"
    assert store.delete(["a"]) == 1
    store.clear()
    assert len(store) == 0


def test_get_vector_store_auto_never_picks_a_network_backend(settings: Settings) -> None:
    store = get_vector_store(settings.model_copy(update={"vector_store": "auto"}), DIM)
    assert store.name in {"numpy", "faiss"}
    store.add(["a"], matrix(0))
    assert store.search(basis(0), 1)[0][0] == "a"


@pytest.mark.skipif(_installed("faiss"), reason="faiss is installed, so no fallback happens")
def test_get_vector_store_falls_back_to_numpy_without_faiss(settings: Settings) -> None:
    store = get_vector_store(settings.model_copy(update={"vector_store": "faiss"}), DIM)
    assert isinstance(store, NumpyVectorStore)


def test_get_vector_store_rejects_an_unknown_backend(settings: Settings) -> None:
    with pytest.raises(ConfigurationError):
        get_vector_store(settings.model_copy(update={"vector_store": "pinecone"}), DIM)


# --------------------------------------------------------------------------- #
# server-backed stores: only the failure modes CI can reach
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_installed("psycopg"), reason="psycopg is installed")
def test_pgvector_without_the_driver_raises_configuration_error(settings: Settings) -> None:
    cfg = settings.model_copy(update={"vector_store": "pgvector"})
    with pytest.raises(ConfigurationError) as excinfo:
        get_vector_store(cfg, DIM)
    assert "psycopg" in str(excinfo.value)


@pytest.mark.skipif(_installed("qdrant_client"), reason="qdrant-client is installed")
def test_qdrant_without_the_driver_raises_configuration_error(settings: Settings) -> None:
    cfg = settings.model_copy(update={"vector_store": "qdrant"})
    with pytest.raises(ConfigurationError) as excinfo:
        get_vector_store(cfg, DIM)
    assert "qdrant_client" in str(excinfo.value)


@pytest.mark.parametrize(
    "table",
    [
        "rag_chunks; DROP TABLE users --",
        'rag_chunks" ; DELETE FROM rag_chunks; --',
        "rag chunks",
        "1_chunks",
        "",
        "x" * 64,
    ],
)
def test_pgvector_rejects_a_table_name_that_is_not_an_identifier(
    settings: Settings, table: str
) -> None:
    cfg = settings.model_copy(update={"vector_store": "pgvector", "pg_table": table})
    with pytest.raises(ConfigurationError) as excinfo:
        PgVectorStore(cfg, DIM)
    # Validation must happen before the driver import, so this is the table
    # error even on a machine where psycopg is installed.
    assert "pg_table" in excinfo.value.message


def test_pgvector_accepts_the_default_table_name(settings: Settings) -> None:
    """The default must pass validation; anything past that needs a live server."""
    from app.retrieval.vectorstore import _validate_identifier

    assert _validate_identifier(settings.pg_table, "pg_table") == settings.pg_table


def test_server_backed_stores_declare_the_full_interface() -> None:
    for cls in (PgVectorStore, QdrantStore):
        assert not getattr(cls, "__abstractmethods__", set()), f"{cls.__name__} is abstract"
