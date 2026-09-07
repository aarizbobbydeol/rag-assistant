"""Dense vector storage behind one small interface.

The default `NumpyVectorStore` keeps every vector in a single float32 matrix and
answers a query with one matmul: exact cosine, no index build, no dependency.
`FaissVectorStore`, `PgVectorStore` and `QdrantStore` swap in for larger corpora
and are imported lazily so a machine without the driver can still import this
module, run the tests and serve traffic on the numpy store.

Every store normalises what it holds, so `search` scores are true cosine in
[-1, 1] regardless of whether the caller handed in unit vectors.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.errors import ConfigurationError
from app.observability import get_logger

logger = get_logger(__name__)

# uuid5 needs a fixed namespace for point ids to be reproducible across
# processes and re-ingests; NAMESPACE_URL plus a project-scoped prefix keeps
# our ids from colliding with anything else in a shared Qdrant instance.
_POINT_NAMESPACE = uuid.NAMESPACE_URL
_POINT_PREFIX = "urn:rag-assistant:chunk:"

# Postgres identifiers are interpolated into DDL, so they are restricted to a
# plain unquoted identifier and never accepted straight from user input.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _as_matrix(vectors: np.ndarray, dim: int, n_ids: int) -> np.ndarray:
    """Coerce input to a validated ``(n_ids, dim)`` float32 matrix."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {array.shape}")
    if array.shape[0] != n_ids:
        raise ValueError(f"got {n_ids} ids but {array.shape[0]} vectors")
    if array.shape[1] != dim:
        raise ValueError(f"expected vectors of dimension {dim}, got {array.shape[1]}")
    return array


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving it at zero makes every cosine
    # against it 0.0, which is the honest answer.
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)


def _query_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != dim:
        raise ValueError(f"expected a query of dimension {dim}, got {array.shape[0]}")
    return _l2_normalise(array.reshape(1, -1))[0]


def _validate_identifier(name: str, kind: str) -> str:
    """Guard against SQL injection through a configured object name."""
    if not _IDENTIFIER_RE.match(name or ""):
        raise ConfigurationError(
            f"invalid {kind} name",
            f"{name!r} is not a plain SQL identifier ([A-Za-z_][A-Za-z0-9_]*, <=63 chars)",
        )
    return name


class VectorStore(ABC):
    """Add / search / delete over unit vectors keyed by chunk id."""

    name: str = "base"

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert ``vectors``; an id that already exists is replaced."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-``k`` ``(id, cosine)`` pairs, descending. Empty store -> ``[]``."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Remove ``ids``; returns how many were actually present."""

    @abstractmethod
    def clear(self) -> None:
        """Drop every vector."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist to ``path`` (a base path; suffixes are added per backend)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore what :meth:`save` wrote to ``path``."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(dim={self.dim}, size={len(self)})"


# --------------------------------------------------------------------------- #
# In-process default
# --------------------------------------------------------------------------- #
class NumpyVectorStore(VectorStore):
    """Exact cosine search over a dense float32 matrix.

    Rows stay contiguous: `delete` compacts the matrix and rebuilds the
    id -> row map, so a deleted vector can never be resurrected by a later
    search or shadow a re-added id.
    """

    name = "numpy"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._ids: list[str] = []
        self._rows: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        if not ids:
            return
        matrix = _l2_normalise(_as_matrix(vectors, self.dim, len(ids)))

        appended: list[np.ndarray] = []
        appended_ids: list[str] = []
        pending: dict[str, int] = {}  # id -> index into ``appended``
        for chunk_id, vector in zip(ids, matrix, strict=False):
            row = self._rows.get(chunk_id)
            if row is not None:
                self._matrix[row] = vector
            elif chunk_id in pending:
                # Duplicated inside this very batch: last write wins, exactly
                # as it would across two separate add() calls.
                appended[pending[chunk_id]] = vector
            else:
                pending[chunk_id] = len(appended)
                appended.append(vector)
                appended_ids.append(chunk_id)

        if appended:
            block = np.asarray(appended, dtype=np.float32)
            self._matrix = np.vstack((self._matrix, block)) if len(self._ids) else block
            for offset, chunk_id in enumerate(appended_ids):
                self._rows[chunk_id] = len(self._ids) + offset
            self._ids.extend(appended_ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        query = _query_vector(vector, self.dim)
        scores = self._matrix @ query
        k = min(k, len(self._ids))
        # A full stable argsort (rather than argpartition) keeps ties broken by
        # insertion order, which the determinism guarantee depends on.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def delete(self, ids: Sequence[str]) -> int:
        targets = {chunk_id for chunk_id in ids if chunk_id in self._rows}
        if not targets:
            return 0
        keep = [row for row, chunk_id in enumerate(self._ids) if chunk_id not in targets]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._ids = [self._ids[row] for row in keep]
        self._rows = {chunk_id: row for row, chunk_id in enumerate(self._ids)}
        return len(targets)

    def clear(self) -> None:
        self._ids = []
        self._rows = {}
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def save(self, path: Path) -> None:
        vectors_path, ids_path = _numpy_paths(path)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        with vectors_path.open("wb") as handle:
            np.savez(handle, vectors=self._matrix)
        ids_path.write_text(
            json.dumps({"dim": self.dim, "ids": self._ids}), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        vectors_path, ids_path = _numpy_paths(path)
        if not vectors_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"no saved vector store at {vectors_path}")
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        stored_dim = int(payload.get("dim", self.dim))
        if stored_dim != self.dim:
            raise ValueError(f"saved store has dim {stored_dim}, this store has {self.dim}")
        with np.load(vectors_path, allow_pickle=False) as archive:
            matrix = np.asarray(archive["vectors"], dtype=np.float32)
        ids = [str(chunk_id) for chunk_id in payload.get("ids", [])]
        if matrix.shape[0] != len(ids) or (matrix.size and matrix.shape[1] != self.dim):
            raise ValueError("saved vectors and id list disagree")
        self._matrix = matrix.reshape(len(ids), self.dim)
        self._ids = ids
        self._rows = {chunk_id: row for row, chunk_id in enumerate(ids)}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        """Insertion-ordered ids; a copy, so callers cannot corrupt the map."""
        return list(self._ids)

    def vectors(self) -> np.ndarray:
        """The normalised matrix, copied - used by MMR and by tests."""
        return self._matrix.copy()


def _numpy_paths(path: Path) -> tuple[Path, Path]:
    """``vectors`` / ``vectors.npz`` both map to the same pair of files."""
    base = Path(path)
    if base.suffix == ".npz":
        base = base.with_suffix("")
    return base.with_name(base.name + ".npz"), base.with_name(base.name + ".ids.json")


# --------------------------------------------------------------------------- #
# Optional backends
# --------------------------------------------------------------------------- #
class FaissVectorStore(VectorStore):
    """`IndexFlatIP` over pre-normalised vectors, so inner product *is* cosine.

    faiss owns no id mapping of its own, so this class keeps the row -> id list
    and mirrors every mutation onto it.
    """

    name = "faiss"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._faiss = _import_or_config_error("faiss", "vector_store='faiss'")
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._ids: list[str] = []
        self._rows: dict[str, int] = {}

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        if not ids:
            return
        matrix = _l2_normalise(_as_matrix(vectors, self.dim, len(ids)))

        replacements: dict[int, np.ndarray] = {}
        appended: list[np.ndarray] = []
        appended_ids: list[str] = []
        pending: dict[str, int] = {}
        for chunk_id, vector in zip(ids, matrix, strict=False):
            row = self._rows.get(chunk_id)
            if row is not None:
                replacements[row] = vector
            elif chunk_id in pending:
                appended[pending[chunk_id]] = vector
            else:
                pending[chunk_id] = len(appended)
                appended.append(vector)
                appended_ids.append(chunk_id)

        if replacements:
            # IndexFlatIP has no in-place update, so rebuild from the stored
            # vectors with the replaced rows patched in.
            existing = self._reconstruct()
            for row, vector in replacements.items():
                existing[row] = vector
            self._index = self._faiss.IndexFlatIP(self.dim)
            self._index.add(existing)
        if appended:
            self._index.add(np.asarray(appended, dtype=np.float32))
            for offset, chunk_id in enumerate(appended_ids):
                self._rows[chunk_id] = len(self._ids) + offset
            self._ids.extend(appended_ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        query = _query_vector(vector, self.dim).reshape(1, -1)
        scores, rows = self._index.search(query, min(k, len(self._ids)))
        return [
            (self._ids[int(row)], float(score))
            for row, score in zip(rows[0], scores[0], strict=False)
            if int(row) >= 0
        ]

    def delete(self, ids: Sequence[str]) -> int:
        targets = {chunk_id for chunk_id in ids if chunk_id in self._rows}
        if not targets:
            return 0
        keep = [row for row, chunk_id in enumerate(self._ids) if chunk_id not in targets]
        # Rebuilding rather than `remove_ids` keeps row order defined by us
        # instead of by faiss' compaction strategy.
        kept = self._reconstruct()[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._index = self._faiss.IndexFlatIP(self.dim)
        if len(kept):
            self._index.add(kept)
        self._ids = [self._ids[row] for row in keep]
        self._rows = {chunk_id: row for row, chunk_id in enumerate(self._ids)}
        return len(targets)

    def clear(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._ids = []
        self._rows = {}

    def save(self, path: Path) -> None:
        index_path, ids_path = _faiss_paths(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        ids_path.write_text(
            json.dumps({"dim": self.dim, "ids": self._ids}), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        index_path, ids_path = _faiss_paths(path)
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"no saved faiss index at {index_path}")
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        if int(payload.get("dim", self.dim)) != self.dim:
            raise ValueError(f"saved index has dim {payload.get('dim')}, expected {self.dim}")
        self._index = self._faiss.read_index(str(index_path))
        self._ids = [str(chunk_id) for chunk_id in payload.get("ids", [])]
        self._rows = {chunk_id: row for row, chunk_id in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)

    def _reconstruct(self) -> np.ndarray:
        total = self._index.ntotal
        if not total:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._index.reconstruct_n(0, total), dtype=np.float32)


def _faiss_paths(path: Path) -> tuple[Path, Path]:
    base = Path(path)
    if base.suffix == ".faiss":
        base = base.with_suffix("")
    return base.with_name(base.name + ".faiss"), base.with_name(base.name + ".ids.json")


class PgVectorStore(VectorStore):
    """pgvector-backed store: the table *is* the index, so save/load are no-ops.

    Every statement is parameterised; the only identifier that reaches the SQL
    text is the configured table name, validated as a plain identifier before
    `psycopg` is even imported.
    """

    name = "pgvector"

    def __init__(self, settings: Settings, dim: int) -> None:
        super().__init__(dim)
        self.table = _validate_identifier(settings.pg_table, "pg_table")
        psycopg = _import_or_config_error("psycopg", "vector_store='pgvector'")
        self._sql = _import_or_config_error("psycopg.sql", "vector_store='pgvector'")
        self._conn = psycopg.connect(settings.pg_dsn, autocommit=True)
        self._table_sql = self._sql.Identifier(self.table)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                self._sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (id text PRIMARY KEY, embedding vector({}))"
                ).format(self._table_sql, self._sql.Literal(self.dim))
            )

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        if not ids:
            return
        matrix = _l2_normalise(_as_matrix(vectors, self.dim, len(ids)))
        rows = [(chunk_id, _pg_vector_literal(vec)) for chunk_id, vec in zip(ids, matrix, strict=False)]
        statement = self._sql.SQL(
            "INSERT INTO {} (id, embedding) VALUES (%s, %s::vector) "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
        ).format(self._table_sql)
        with self._conn.cursor() as cur:
            cur.executemany(statement, rows)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        literal = _pg_vector_literal(_query_vector(vector, self.dim))
        statement = self._sql.SQL(
            "SELECT id, embedding <=> %s::vector AS distance FROM {} "
            "ORDER BY embedding <=> %s::vector ASC LIMIT %s"
        ).format(self._table_sql)
        with self._conn.cursor() as cur:
            cur.execute(statement, (literal, literal, int(k)))
            # `<=>` is cosine *distance*; the contract is cosine similarity.
            return [(str(row[0]), 1.0 - float(row[1])) for row in cur.fetchall()]

    def delete(self, ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return 0
        statement = self._sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(self._table_sql)
        with self._conn.cursor() as cur:
            cur.execute(statement, (ids,))
            return int(cur.rowcount or 0)

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._sql.SQL("TRUNCATE TABLE {}").format(self._table_sql))

    def save(self, path: Path) -> None:
        """No-op: Postgres already persists the table."""

    def load(self, path: Path) -> None:
        """No-op: the table is read in place."""

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(self._sql.SQL("SELECT count(*) FROM {}").format(self._table_sql))
            row = cur.fetchone()
            return int(row[0]) if row else 0


def _pg_vector_literal(vector: np.ndarray) -> str:
    """pgvector's text input format, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class QdrantStore(VectorStore):
    """Qdrant collection with COSINE distance; point ids are uuid5 of the chunk id.

    Qdrant only accepts unsigned integers or UUIDs as point ids, so the chunk id
    is hashed into a namespaced uuid5 - deterministic, so re-ingesting the same
    corpus updates points instead of duplicating them - and kept verbatim in the
    payload, which is what `search` returns.
    """

    name = "qdrant"

    def __init__(self, settings: Settings, dim: int) -> None:
        super().__init__(dim)
        client_mod = _import_or_config_error("qdrant_client", "vector_store='qdrant'")
        self._models = _import_or_config_error("qdrant_client.models", "vector_store='qdrant'")
        self.collection = settings.qdrant_collection
        self._client = client_mod.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=self._models.VectorParams(
                size=self.dim, distance=self._models.Distance.COSINE
            ),
        )

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        if not ids:
            return
        matrix = _l2_normalise(_as_matrix(vectors, self.dim, len(ids)))
        points = [
            self._models.PointStruct(
                id=_point_id(chunk_id),
                vector=[float(value) for value in vector],
                payload={"chunk_id": chunk_id},
            )
            for chunk_id, vector in zip(ids, matrix, strict=False)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = [float(value) for value in _query_vector(vector, self.dim)]
        query_points = getattr(self._client, "query_points", None)
        if query_points is not None:
            hits = query_points(
                collection_name=self.collection, query=query, limit=int(k), with_payload=True
            ).points
        else:  # qdrant-client < 1.10 predates the query API
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=query,
                limit=int(k),
                with_payload=True,
            )
        return [
            (str((hit.payload or {}).get("chunk_id", hit.id)), float(hit.score))
            for hit in hits
        ]

    def delete(self, ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return 0
        point_ids = [_point_id(chunk_id) for chunk_id in ids]
        # Qdrant's delete is idempotent and reports no count, so ask what is
        # actually there first to keep the documented return value honest.
        found = self._client.retrieve(
            collection_name=self.collection, ids=point_ids, with_payload=False
        )
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(found)

    def clear(self) -> None:
        self._client.delete_collection(collection_name=self.collection)
        self._ensure_collection()

    def save(self, path: Path) -> None:
        """No-op: Qdrant persists the collection itself."""

    def load(self, path: Path) -> None:
        """No-op: the collection is queried in place."""

    def __len__(self) -> int:
        return int(self._client.count(collection_name=self.collection, exact=True).count)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, _POINT_PREFIX + chunk_id))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _import_or_config_error(module: str, requested: str) -> Any:
    """Import an optional driver, turning ImportError into ConfigurationError.

    Callers configure a backend; a missing driver is a configuration problem,
    not an unhandled import, and the API layer already maps
    :class:`ConfigurationError` onto a 500 with a useful message.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        package = module.split(".")[0]
        raise ConfigurationError(
            f"{requested} requires the '{package}' package",
            f"pip install {package}",
        ) from exc


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def get_vector_store(settings: Settings, dim: int) -> VectorStore:
    """Build the store named by ``settings.vector_store``.

    ``auto`` resolves faiss -> numpy and never reaches for a network store: a
    silently-chosen Postgres or Qdrant would turn a missing service into a
    startup failure. faiss falls back to numpy even when named explicitly,
    because `IndexFlatIP` and the numpy matmul return identical results; the
    server backends raise instead, since falling back there would quietly write
    to the wrong place.
    """
    backend = (settings.vector_store or "auto").strip().lower()

    if backend in {"auto", ""}:
        backend = "faiss" if _module_available("faiss") else "numpy"

    if backend == "numpy":
        return NumpyVectorStore(dim)
    if backend == "faiss":
        try:
            return FaissVectorStore(dim)
        except ConfigurationError:
            logger.warning("faiss_unavailable_using_numpy", extra={"dim": dim})
            return NumpyVectorStore(dim)
    if backend in {"pgvector", "pg", "postgres", "postgresql"}:
        return PgVectorStore(settings, dim)
    if backend == "qdrant":
        return QdrantStore(settings, dim)
    raise ConfigurationError(
        f"unknown vector_store {settings.vector_store!r}",
        "expected one of: auto, numpy, faiss, pgvector, qdrant",
    )


__all__ = [
    "VectorStore",
    "NumpyVectorStore",
    "FaissVectorStore",
    "PgVectorStore",
    "QdrantStore",
    "get_vector_store",
]
