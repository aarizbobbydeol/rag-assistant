"""The golden set: a JSONL file of questions with the passages that answer them.

The important design decision lives in :func:`resolve_relevant_chunks`. A golden
set that pinned relevance to ``chunk_id`` would be invalidated by every sweep
configuration, because changing the splitter or the chunk size changes every id.
So the ground truth is stored as *text* - the passage a human judged relevant -
and resolved against whatever chunks are actually in the index at scoring time.
That is what lets one dataset grade `recursive/700` against `semantic/300`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.errors import ConfigurationError
from app.models import Chunk
from app.utils import containment, normalize_whitespace, tokenize

__all__ = [
    "EvalExample",
    "load_dataset",
    "save_dataset",
    "resolve_relevant_chunks",
]

# A passage that straddles a chunk boundary is in no single chunk in full, so a
# chunk carrying at least this share of its words is accepted as a partial hit.
_PARTIAL_MATCH_THRESHOLD = 0.6


@dataclass
class EvalExample:
    """One graded question.

    ``relevant_doc_ids`` and ``relevant_chunk_texts`` are both required by the
    module contract; either may be empty, and :meth:`from_dict` fills in the
    ones a hand-written JSONL line leaves out. ``must_abstain`` marks the
    questions the corpus deliberately cannot answer - the harness scores those
    on refusal, not on answer overlap.
    """

    question: str
    answer: str
    relevant_doc_ids: list[str]
    relevant_chunk_texts: list[str]
    must_abstain: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalExample:
        """Build an example from a decoded JSONL object, tolerating omissions.

        Raises ``KeyError`` for a missing ``question``; :func:`load_dataset`
        turns that into an error naming the offending line.
        """
        if "question" not in data:
            raise KeyError("question")
        return cls(
            question=str(data["question"]),
            answer=str(data.get("answer", "")),
            relevant_doc_ids=[str(item) for item in data.get("relevant_doc_ids", [])],
            relevant_chunk_texts=[str(item) for item in data.get("relevant_chunk_texts", [])],
            must_abstain=bool(data.get("must_abstain", False)),
            tags=[str(item) for item in data.get("tags", [])],
        )


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Read a JSONL golden set, skipping blank lines.

    Malformed lines raise :class:`app.errors.ConfigurationError` naming the file
    and the 1-based line number, because "Expecting ',' delimiter: line 1
    column 84" on its own is useless when the dataset has two hundred rows.
    """
    source = Path(path)
    examples: list[EvalExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"{source}: line {line_no} is not valid JSON",
                    detail=str(exc),
                ) from exc
            if not isinstance(payload, dict):
                raise ConfigurationError(
                    f"{source}: line {line_no} is not a JSON object",
                    detail=f"got {type(payload).__name__}",
                )
            try:
                examples.append(EvalExample.from_dict(payload))
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{source}: line {line_no} is not a valid eval example",
                    detail=f"missing or malformed field: {exc}",
                ) from exc
    return examples


def save_dataset(path: str | Path, examples: Iterable[EvalExample]) -> int:
    """Write examples as JSONL, creating parent directories. Returns the count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def _canonical(text: str) -> str:
    """Whitespace-, case- and punctuation-insensitive form used for matching.

    Both sides of every comparison go through this, so a passage recorded with
    markdown bullets still matches the same passage after a splitter stripped
    the newlines.
    """
    return " ".join(tokenize(normalize_whitespace(text)))


def resolve_relevant_chunks(example: EvalExample, chunks: Sequence[Chunk]) -> set[str]:
    """Map an example's ground truth onto the chunk ids present in the index.

    A chunk counts as relevant when, after canonicalisation, it *contains* a
    recorded passage or is *contained by* one - the second direction matters
    because a finer splitter turns one recorded passage into several chunks that
    are each individually relevant. Passages that no chunk contains outright
    fall back to a word-overlap test, which recovers the case where the passage
    straddles a chunk boundary.

    When no passage matches anything - an empty ``relevant_chunk_texts``, or a
    corpus edited since the golden set was written - relevance falls back to
    every chunk of a relevant document, so the example still grades something.
    """
    canonical_chunks = [(chunk.chunk_id, _canonical(chunk.text)) for chunk in chunks]
    resolved: set[str] = set()

    for passage in example.relevant_chunk_texts:
        needle = _canonical(passage)
        if not needle:
            continue
        hits = {
            chunk_id
            for chunk_id, haystack in canonical_chunks
            if haystack and (needle in haystack or haystack in needle)
        }
        if not hits:
            hits = _partial_hits(needle, canonical_chunks)
        resolved |= hits

    if not resolved and example.relevant_doc_ids:
        relevant_docs = set(example.relevant_doc_ids)
        resolved = {chunk.chunk_id for chunk in chunks if chunk.doc_id in relevant_docs}
    return resolved


def _partial_hits(needle: str, canonical_chunks: Sequence[tuple[str, str]]) -> set[str]:
    """Chunks sharing at least ``_PARTIAL_MATCH_THRESHOLD`` of ``needle``'s words.

    Containment is measured in both directions so that a passage split across
    two chunks and a chunk swallowed whole by a passage both register.
    """
    needle_tokens = set(needle.split())
    if not needle_tokens:
        return set()
    hits: set[str] = set()
    for chunk_id, haystack in canonical_chunks:
        chunk_tokens = set(haystack.split())
        if not chunk_tokens:
            continue
        covered = max(
            containment(needle_tokens, chunk_tokens),
            containment(chunk_tokens, needle_tokens),
        )
        if covered >= _PARTIAL_MATCH_THRESHOLD:
            hits.add(chunk_id)
    return hits
