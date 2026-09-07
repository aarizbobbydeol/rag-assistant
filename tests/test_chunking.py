"""Invariant tests for the four splitters.

The invariants below are what the rest of the pipeline is allowed to assume, so
they are asserted for *every* registered splitter rather than for one of them:

* ``doc.text[c.start_char:c.end_char] == c.text`` - exact offsets into the source
* ordinals are 0-based and contiguous, chunk ids unique and process-stable
* ``estimate_tokens(c.text) <= chunk_size + min_chunk_tokens`` - the documented
  oversize tolerance, one merged-in undersized neighbour
* consecutive chunks share text when ``chunk_overlap > 0`` and are disjoint when
  it is 0
* nothing below ``min_chunk_tokens`` survives except a whole tiny document
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.errors import ConfigurationError
from app.ingestion.chunking import (
    SPLITTERS,
    FixedSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    SentenceSplitter,
    Splitter,
    get_splitter,
)
from app.models import Chunk, Document, SplitterName
from app.utils import estimate_tokens, split_sentences

NAMES = sorted(SPLITTERS)

CHUNK_SIZE = 48
CHUNK_OVERLAP = 12
MIN_TOKENS = 8

LONG_TEXT = """# Support handbook

The support desk answers billing questions during business hours. Agents record
every billing question in the ticket system before replying. A billing question
that needs a refund is escalated to the finance team.

## Refunds

Customers may request a refund within 30 days of purchase. Refunds are issued to
the original payment method within ten business days. A refund request outside
the 30 day window is declined automatically.

## Shipping

Standard shipping takes four to six business days. Express shipping arrives the
next business day when the order is placed before noon. Shipping fees are never
refunded once the parcel has left the warehouse.

## Accounts

An account can be closed from the settings page at any time. Closing an account
deletes the stored payment methods after thirty days. An account that owes money
cannot be closed until the balance is settled.
"""

TINY_TEXT = "One short line."


def make_doc(text: str, doc_id: str = "doc-handbook", **metadata: object) -> Document:
    return Document(
        doc_id=doc_id,
        source="handbook.md",
        title="Support handbook",
        text=text,
        metadata=dict(metadata),
    )


def build(
    name: str,
    text: str = LONG_TEXT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_tokens: int = MIN_TOKENS,
) -> tuple[Document, list[Chunk]]:
    doc = make_doc(text)
    splitter = get_splitter(name, chunk_size, chunk_overlap, min_chunk_tokens)
    return doc, splitter.split(doc)


def covered_indices(chunks: list[Chunk]) -> set[int]:
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_char, chunk.end_char))
    return covered


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_holds_the_four_contract_splitters():
    assert set(SPLITTERS) == {"recursive", "fixed", "sentence", "semantic"}
    assert SPLITTERS["recursive"] is RecursiveSplitter
    assert SPLITTERS["fixed"] is FixedSplitter
    assert SPLITTERS["sentence"] is SentenceSplitter
    assert SPLITTERS["semantic"] is SemanticSplitter
    for name, cls in SPLITTERS.items():
        assert issubclass(cls, Splitter)
        assert cls.name == name


@pytest.mark.parametrize("name", NAMES)
def test_get_splitter_accepts_str_and_enum(name: str):
    by_str = get_splitter(name, 100, 10, 5)
    by_enum = get_splitter(SplitterName(name), 100, 10, 5)
    assert type(by_str) is type(by_enum) is SPLITTERS[name]
    assert (by_str.chunk_size, by_str.chunk_overlap, by_str.min_chunk_tokens) == (100, 10, 5)


def test_get_splitter_defaults_and_settings_roundtrip():
    settings = Settings()
    splitter = get_splitter(
        settings.splitter, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_tokens
    )
    assert isinstance(splitter, RecursiveSplitter)
    assert get_splitter("recursive", 100, 10).min_chunk_tokens == 24


def test_get_splitter_rejects_unknown_name():
    with pytest.raises(ConfigurationError):
        get_splitter("word2vec", 100, 10)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 250)],
)
def test_invalid_sizes_are_configuration_errors(chunk_size: int, chunk_overlap: int):
    with pytest.raises(ConfigurationError):
        get_splitter("recursive", chunk_size, chunk_overlap)


# --------------------------------------------------------------------------- #
# Shared invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_offsets_index_back_into_the_document(name: str):
    doc, chunks = build(name)
    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize("name", NAMES)
def test_ordinals_are_zero_based_and_contiguous(name: str):
    _, chunks = build(name)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("name", NAMES)
def test_chunk_ids_unique_and_stable_across_runs(name: str):
    _, first = build(name)
    _, second = build(name)
    ids = [c.chunk_id for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    assert [(c.start_char, c.end_char) for c in first] == [
        (c.start_char, c.end_char) for c in second
    ]


@pytest.mark.parametrize("name", NAMES)
def test_chunk_ids_differ_between_documents(name: str):
    splitter = get_splitter(name, CHUNK_SIZE, CHUNK_OVERLAP, MIN_TOKENS)
    left = splitter.split(make_doc(LONG_TEXT, doc_id="doc-a"))
    right = splitter.split(make_doc(LONG_TEXT, doc_id="doc-b"))
    assert {c.chunk_id for c in left}.isdisjoint({c.chunk_id for c in right})


@pytest.mark.parametrize("name", NAMES)
def test_no_chunk_exceeds_the_documented_tolerance(name: str):
    # Tolerance: chunk_size + min_chunk_tokens. A chunk is packed to at most
    # chunk_size tokens (chunk_size - chunk_overlap of new material plus the
    # overlap glued on); the only way past that is absorbing one undersized
    # neighbour, which is bounded by min_chunk_tokens.
    limit = CHUNK_SIZE + MIN_TOKENS
    _, chunks = build(name)
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= limit
        assert chunk.token_count == estimate_tokens(chunk.text)


@pytest.mark.parametrize("name", NAMES)
def test_undersized_chunks_are_merged_away(name: str):
    _, chunks = build(name)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count >= MIN_TOKENS


@pytest.mark.parametrize("name", NAMES)
def test_trailing_fragment_does_not_survive(name: str):
    # A stub sentence after a full-sized body is the classic tiny-tail case.
    text = LONG_TEXT + "\n\n## Note\n\nSee above."
    _, chunks = build(name, text=text)
    assert chunks[-1].token_count >= MIN_TOKENS
    assert "See above." in chunks[-1].text


@pytest.mark.parametrize("name", NAMES)
def test_chunks_are_ordered_trimmed_and_cover_every_word(name: str):
    doc, chunks = build(name)
    previous_start = -1
    for chunk in chunks:
        assert chunk.start_char > previous_start
        assert chunk.start_char < chunk.end_char
        previous_start = chunk.start_char
        assert chunk.text == chunk.text.strip()

    covered = covered_indices(chunks)
    missed = [i for i, ch in enumerate(doc.text) if not ch.isspace() and i not in covered]
    assert missed == [], f"{name} dropped characters at {missed[:10]}"


@pytest.mark.parametrize("name", NAMES)
def test_overlap_actually_overlaps(name: str):
    doc, chunks = build(name, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) >= 3
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char < previous.end_char
        shared = doc.text[current.start_char : previous.end_char]
        assert shared.strip()
        assert shared in previous.text
        assert current.text.startswith(shared)


@pytest.mark.parametrize("name", NAMES)
def test_zero_overlap_produces_disjoint_chunks(name: str):
    _, chunks = build(name, chunk_overlap=0)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char >= previous.end_char


@pytest.mark.parametrize("name", NAMES)
def test_empty_and_blank_documents_yield_nothing(name: str):
    splitter = get_splitter(name, CHUNK_SIZE, CHUNK_OVERLAP, MIN_TOKENS)
    assert splitter.split(make_doc("")) == []
    assert splitter.split(make_doc("   \n\n\t  ")) == []


@pytest.mark.parametrize("name", NAMES)
def test_document_smaller_than_the_minimum_still_yields_one_chunk(name: str):
    doc, chunks = build(name, text=TINY_TEXT, min_chunk_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text == TINY_TEXT
    assert doc.text[chunks[0].start_char : chunks[0].end_char] == TINY_TEXT


@pytest.mark.parametrize("name", NAMES)
def test_metadata_records_the_splitter(name: str):
    _, chunks = build(name)
    assert {c.metadata["splitter"] for c in chunks} == {name}


@pytest.mark.parametrize("name", NAMES)
def test_text_without_any_separator_stays_bounded(name: str):
    # A single unbroken run has no separator to cut on; the splitter must still
    # terminate and keep the offset invariant.
    doc, chunks = build(name, text="x" * 4000)
    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
    assert "".join(c.text for c in chunks).count("x") >= 4000


@pytest.mark.parametrize("name", NAMES)
def test_conftest_sample_document(name: str, sample_document: Document):
    splitter = get_splitter(name, 32, 8, 6)
    chunks = splitter.split(sample_document)
    assert len(chunks) > 1
    for chunk in chunks:
        assert sample_document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.doc_id == sample_document.doc_id
        assert chunk.source == sample_document.source
        assert chunk.title == sample_document.title
        assert chunk.page is None
        assert estimate_tokens(chunk.text) <= 32 + 6


# --------------------------------------------------------------------------- #
# Page attribution
# --------------------------------------------------------------------------- #
def page_document() -> tuple[Document, list[list[int]]]:
    page_one = (
        "Chapter one covers the intake process. The intake process starts when a "
        "ticket arrives in the queue. Every ticket in the queue is triaged by an "
        "agent before it is assigned. Intake closes at six in the evening on "
        "working days. Tickets that arrive after intake closes wait until the "
        "following morning.\n\n"
    )
    page_two = (
        "Chapter two covers escalation. An escalation is raised when the agent "
        "cannot resolve the ticket. Escalated tickets are reviewed by a lead "
        "every morning. A lead may hand the escalation back with written notes. "
        "Escalations that stay open for a week are reported to the manager.\n\n"
    )
    page_three = (
        "Chapter three covers reporting. Reporting runs weekly over the resolved "
        "tickets. The weekly report is shared with the whole support team. Each "
        "report counts the tickets closed and the escalations raised. Reports "
        "older than a year are archived and then deleted."
    )
    text = page_one + page_two + page_three
    spans = [
        [0, len(page_one), 1],
        [len(page_one), len(page_one) + len(page_two), 2],
        [len(page_one) + len(page_two), len(text), 3],
    ]
    return make_doc(text, doc_id="doc-pdf", page_spans=spans), spans


@pytest.mark.parametrize("name", NAMES)
def test_page_attribution_follows_page_spans(name: str):
    doc, spans = page_document()
    splitter = get_splitter(name, 24, 0, 6)
    chunks = splitter.split(doc)
    assert len(chunks) >= 6

    def expected(pos: int) -> int:
        return next(page for start, end, page in spans if start <= pos < end)

    for chunk in chunks:
        assert chunk.page == expected(chunk.start_char)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert chunks[0].locator().endswith("(p.1)")


@pytest.mark.parametrize("name", NAMES)
def test_page_is_none_without_page_spans(name: str):
    doc, _ = page_document()
    plain = make_doc(doc.text)
    chunks = get_splitter(name, 40, 0, 8).split(plain)
    assert all(c.page is None for c in chunks)


def test_malformed_page_spans_are_ignored_not_fatal():
    doc = make_doc(LONG_TEXT, page_spans=[["nope"], None, [0, 10, 1]])
    chunks = get_splitter("recursive", CHUNK_SIZE, CHUNK_OVERLAP, MIN_TOKENS).split(doc)
    assert chunks
    assert all(c.page == 1 for c in chunks)


# --------------------------------------------------------------------------- #
# Per-splitter behaviour
# --------------------------------------------------------------------------- #
def test_recursive_breaks_on_the_coarsest_structure_that_fits():
    text = (
        "## Alpha\n\nAlpha covers the first topic in two short lines.\n\n"
        "## Beta\n\nBeta covers the second topic in two short lines.\n\n"
        "## Gamma\n\nGamma covers the third topic in two short lines."
    )
    doc = make_doc(text)
    chunks = RecursiveSplitter(chunk_size=24, chunk_overlap=0, min_chunk_tokens=4).split(doc)
    assert [c.text.splitlines()[0] for c in chunks] == ["## Alpha", "## Beta", "## Gamma"]


def test_recursive_descends_when_a_section_is_too_large():
    section = "## Alpha\n\n" + " ".join(f"word{i}" for i in range(200))
    doc = make_doc(section)
    chunks = RecursiveSplitter(chunk_size=40, chunk_overlap=0, min_chunk_tokens=4).split(doc)
    assert len(chunks) > 3
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
        assert estimate_tokens(chunk.text) <= 40 + 4


def test_fixed_windows_are_full_and_regularly_spaced():
    doc = make_doc(LONG_TEXT)
    chunks = FixedSplitter(chunk_size=40, chunk_overlap=10, min_chunk_tokens=4).split(doc)
    assert len(chunks) > 3
    # Every window except the last is filled to the budget rather than to a
    # structural boundary - that is the whole point of the fixed splitter.
    for chunk in chunks[:-1]:
        assert 30 <= chunk.token_count <= 40
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)


def test_sentence_splitter_never_cuts_a_sentence():
    doc = make_doc(LONG_TEXT)
    chunks = SentenceSplitter(chunk_size=48, chunk_overlap=0, min_chunk_tokens=8).split(doc)
    sentences = split_sentences(LONG_TEXT)
    starts = {LONG_TEXT.find(s): s for s in sentences}
    for chunk in chunks:
        assert chunk.start_char in starts
        assert chunk.text.endswith(tuple(".!?")) or chunk.text.splitlines()[-1].startswith("#")


def test_semantic_splitter_breaks_where_the_vocabulary_turns_over():
    refunds = (
        "The refund window is thirty days. The refund window applies to every "
        "order. A refund request inside the refund window is approved."
    )
    scheduling = (
        "Kubernetes schedules pods onto nodes. Pods on nodes are restarted by "
        "kubernetes whenever nodes fail."
    )
    doc = make_doc(refunds + " " + scheduling)
    chunks = SemanticSplitter(chunk_size=400, chunk_overlap=0, min_chunk_tokens=1).split(doc)

    assert len(chunks) == 2
    assert "refund" in chunks[0].text and "Kubernetes" not in chunks[0].text
    assert "Kubernetes" in chunks[1].text and "refund" not in chunks[1].text
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_semantic_threshold_is_tunable():
    doc = make_doc(
        "The refund window is thirty days. The refund window applies to every order."
    )
    # A threshold above the pair's actual similarity forces a break; below it,
    # the two sentences stay together.
    together = SemanticSplitter(400, 0, 1, similarity_threshold=0.05).split(doc)
    apart = SemanticSplitter(400, 0, 1, similarity_threshold=0.95).split(doc)
    assert len(together) == 1
    assert len(apart) == 2


def test_semantic_defaults_match_the_factory_signature():
    splitter = get_splitter("semantic", 100, 10, 5)
    assert isinstance(splitter, SemanticSplitter)
    assert 0.0 < splitter.similarity_threshold < 1.0
