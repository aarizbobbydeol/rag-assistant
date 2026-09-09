"""Tests for anchor attestation: the guard against confidently answering a
question whose subject the corpus never actually mentions.

The boundary cases matter more than the catches here. Every detector trades
against over-refusal, and the golden set's eight must-abstain examples cannot
measure over-refusal at all - so the near-miss questions that must *keep*
working are asserted explicitly, as the regression net.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.generation.anchors import (
    build_anchor_index,
    chunk_vocab,
    missing_anchors,
)

# Miniature stand-ins for the real corpus, carrying only the traps.
RUNBOOK = (
    "Northwind uses three severity levels. There is no SEV-4; anything smaller "
    "than a SEV-3 is filed as a normal defect in the backlog. "
    "The acknowledgement target for SEV-2 is 15 minutes. "
    "The acknowledgement target for SEV-3 is 4 business hours."
)
API = (
    "The Free tier allows 60 requests per minute, the Standard tier 600 "
    "requests per minute, and the Enterprise tier 4,000 requests per minute. "
    "Requests to the v1 base URL are refused; v1 was retired. "
    "Traffic to the Atlas API must use TLS 1.2 or TLS 1.3. "
    "Responses are paginated with page_size defaulting to 50."
)
HANDBOOK = (
    "Any item above 5,000 US dollars requires approval from the Chief "
    "Financial Officer, Marguerite Adeyemi. Access is approved by the Data "
    "Governance Lead and the Chief Information Security Officer."
)
BENEFITS = (
    "Dental coverage is provided through Larkspur Dental and vision through "
    "Clearline Vision. Both are included at no employee premium."
)
# The orphan-span detector only fires on terms the corpus uses *somewhere* -
# otherwise the miss is retrieval's problem, not an anchor's. This chunk is what
# makes "annual maximum" a genuine orphan in the dental passage rather than
# vocabulary the corpus simply does not have.
EXPENSES = (
    "The annual learning budget is 1,500 US dollars per employee. Meals are "
    "reimbursed up to a maximum of 65 US dollars per day."
)

CORPUS = [RUNBOOK, API, HANDBOOK, BENEFITS, EXPENSES]


@pytest.fixture(scope="module")
def index():
    return build_anchor_index(CORPUS)


@pytest.fixture(scope="module")
def vocabs():
    return [chunk_vocab(text) for text in CORPUS]


def fires(question, index, vocabs, **kw):
    return bool(missing_anchors(question, vocabs, index, **kw))


# --------------------------------------------------------------------------- #
# attestation is negation-scoped
# --------------------------------------------------------------------------- #
def test_denied_identifier_is_not_attested(index):
    """'There is no SEV-4' mentions SEV-4 without attesting it.

    A plain presence test passes here, which is exactly why the guard needs
    negation scoping rather than a substring check.
    """
    assert ("sev", "4") not in index.identifiers
    assert ("sev", "3") in index.identifiers


def test_deprecation_is_not_negation(index):
    """'v1 was retired' still attests v1 - the corpus does describe it."""
    assert ("v", "1") in index.identifiers


def test_negation_does_not_leak_across_a_clause_boundary():
    """A cue before a full stop must not suppress the next sentence."""
    index = build_anchor_index(["There is no backlog. SEV-9 is acknowledged in 5 minutes."])
    assert ("sev", "9") in index.identifiers


def test_spaced_identifier_needs_a_learned_family(index):
    """'TLS 1.2' attests; 'within 5 minutes' must never mint a ("within", "5")."""
    assert ("tls", "1.2") in index.identifiers
    assert "within" not in index.families


# --------------------------------------------------------------------------- #
# the catches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "What is the acknowledgement target for a SEV-4 incident?",
        "Who is the Chief Executive Officer of Northwind Cartography?",
        "What is the annual maximum benefit on the Larkspur Dental plan?",
    ],
)
def test_unattested_anchor_fires(question, index, vocabs):
    assert fires(question, index, vocabs)


def test_unknown_version_in_a_known_family_fires(index, vocabs):
    assert fires("Is TLS 1.4 accepted at the edge?", index, vocabs)


# --------------------------------------------------------------------------- #
# the boundary locks - these are what stop the guard eating good questions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        # Relaxed name-phrase matching: the run is not attested end to end, but
        # an adjacent pair is, so these stay answerable.
        "What is the Atlas API Free tier rate limit?",
        "What are the Atlas API Standard tier rate limits?",
        # A real officer, whose every adjacent pair the corpus writes.
        "Who is the Chief Financial Officer?",
        # Attested identifiers, including one ending a sentence.
        "What is the acknowledgement target for a SEV-3 incident?",
        "Is TLS 1.2 still accepted?",
        # page_size is one token in the corpus; sub-token expansion keeps this
        # from looking like an absent anchor.
        "What is the default page size for list endpoints?",
    ],
)
def test_answerable_near_misses_do_not_fire(question, index, vocabs):
    assert not fires(question, index, vocabs)


def test_case_shifted_questions_do_not_fire(index, vocabs):
    """Title Case and ALL CAPS capitalise words the writer never meant as names.

    Reading those as name phrases would refuse a question the corpus answers,
    so the detector stands down when the casing carries no signal.
    """
    question = "What is the acknowledgement target for a SEV-3 incident?"
    for variant in (question.title(), question.upper()):
        assert not fires(variant, index, vocabs)


# --------------------------------------------------------------------------- #
# tokenizer behaviour the detectors rely on
# --------------------------------------------------------------------------- #
def test_chunk_vocab_expands_compound_tokens():
    """``page_size`` is one token, so its parts have to be indexed too.

    Without this the orphan-span detector reads "page size" as two terms the
    chunk never mentions and refuses a question the chunk answers.
    """
    vocab = chunk_vocab("page_size defaults to 50")
    assert {"pag", "siz"} <= vocab


def test_empty_inputs_are_inert(index):
    assert missing_anchors("", [], index) == []
    assert missing_anchors("anything", [], build_anchor_index([])) == []


# --------------------------------------------------------------------------- #
# the property that actually protects the 58 answerable questions
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden" / "qa.jsonl"
CORPUS_DIR = ROOT / "data" / "corpus"


def _answerable_questions() -> list[str]:
    if not GOLDEN.exists():  # pragma: no cover - dataset ships with the repo
        return []
    rows = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row["question"] for row in rows if not row.get("must_abstain")]


@pytest.fixture(scope="module")
def real_corpus():
    """The shipped markdown corpus, read as plain text.

    The PDF needs a loader and contributes no anchors these tests turn on, so
    the markdown alone is enough to make this a real regression net.
    """
    texts = [path.read_text(encoding="utf-8") for path in sorted(CORPUS_DIR.glob("*.md"))]
    if not texts:  # pragma: no cover - corpus ships with the repo
        pytest.skip("corpus not present")
    return build_anchor_index(texts), [chunk_vocab(text) for text in texts]


@pytest.mark.parametrize("question", _answerable_questions())
def test_no_answerable_question_fires_in_any_casing(question, real_corpus):
    """The guard must never refuse a question the corpus answers.

    Eight must-abstain examples cannot measure over-refusal, so this is the file
    that actually protects the other 58. Casing is included because it is a
    signal the writer controls and the reader should not depend on: a guard that
    refuses a lowercased question but answers the Title Case one is reading the
    keyboard, not the corpus.
    """
    index, vocabs = real_corpus
    for variant in (question, question.lower(), question.title(), question.upper()):
        assert missing_anchors(variant, vocabs, index) == [], f"fired on: {variant}"


# --------------------------------------------------------------------------- #
# regressions found by the held-out set
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        # Ordinary verb and adverb phrases are not missing attributes. Each of
        # these wrongly refused a question the corpus answers, because the span
        # was two adjacent content words the pivot chunk happened not to use.
        "How long will the old signing secret keep working?",
        "Is TLS 1.2 still acceptable for data in transit?",
        "When did that version take effect?",
    ],
)
def test_orphan_span_ignores_ordinary_verb_phrases(question, index, vocabs):
    assert not fires(question, index, vocabs)


def test_orphan_span_still_catches_a_missing_attribute(index, vocabs):
    """The detector's actual job, which the verb-phrase fix must not break."""
    misses = missing_anchors(
        "What is the annual maximum benefit on the Larkspur Dental plan?", vocabs, index
    )
    assert [m.detector for m in misses] == ["orphan_span"]


def test_version_prefix_aliases_fold_together(index, vocabs):
    """The header says "Version 2.8"; readers ask about "v2.8"."""
    corpus = build_anchor_index(["Atlas API. Version 2.8 | Effective 1 March 2025"])
    assert ("v", "2.8") in corpus.identifiers
    assert not missing_anchors("When did v2.8 take effect?", vocabs, corpus)


# --------------------------------------------------------------------------- #
# code identifiers - the vocabulary of a software-docs corpus
# --------------------------------------------------------------------------- #
DOCS = (
    "Use `UploadFile` for large uploads. Register a dependency with `Depends`. "
    "Run `uvicorn main:app --reload`. The `response_model` argument takes priority. "
    "You can use app.mount to add a sub-application. Traffic must use TLS 1.2."
)


@pytest.fixture(scope="module")
def code_index():
    return build_anchor_index([DOCS])


@pytest.mark.parametrize(
    "question",
    [
        "How do I register a dependency with @app.dependency()?",   # decorator
        "What does the `anyio_backend` fixture return?",            # snake_case
        "What is `ACCESS_TOKEN_EXPIRE_MINUTES` set to?",            # SCREAMING_SNAKE
        "What is the default for --timeout-keep-alive?",            # cli flag
        "When does `jwt.decode()` raise?",                          # dotted, in code font
    ],
)
def test_invented_code_identifiers_fire(question, code_index):
    """These are the names a software corpus is asked about and never contains.

    The other detectors are blind to them: name_phrase keys on capitalisation
    and stands down when casing carries no signal, which for code is always,
    and identifier requires a digit.
    """
    misses = missing_anchors(question, [chunk_vocab(DOCS)], code_index)
    assert any(m.detector == "code_identifier" for m in misses), question


@pytest.mark.parametrize(
    "question",
    [
        "How do I use `UploadFile` for a large upload?",
        "What does the `response_model` argument do?",
        "How do I declare a dependency with `Depends`?",
    ],
)
def test_attested_code_identifiers_do_not_fire(question, code_index):
    assert not missing_anchors(question, [chunk_vocab(DOCS)], code_index)


def test_a_dotted_path_in_plain_prose_is_not_an_anchor(code_index):
    """People write "app.mount" mid-sentence to mean mounting, not to cite an API.

    Documentation that explains mounting in prose while keeping the code in
    separate files never contains the literal string, so treating a bare dotted
    path as a claim refuses questions the docs answer.
    """
    question = "When I mount a sub-application with app.mount(), what happens to /docs?"
    assert not [m for m in missing_anchors(question, [chunk_vocab(DOCS)], code_index)
                if m.detector == "code_identifier"]


def test_the_detector_stays_silent_on_a_prose_corpus(index, vocabs):
    """A policy corpus has no code, so this detector must never fire on one."""
    for question in (
        "How much notice is required before taking leave?",
        "Who approves an expense above five thousand dollars?",
    ):
        assert not [m for m in missing_anchors(question, vocabs, index)
                    if m.detector == "code_identifier"]
