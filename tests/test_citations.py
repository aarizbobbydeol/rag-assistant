"""Tests for `[n]` marker parsing, resolution and renumbering."""

from __future__ import annotations

import pytest

from app.generation.citations import (
    QUOTE_MAX_CHARS,
    extract_markers,
    renumber_answer,
    resolve_citations,
    strip_markers,
)

REFUNDS = (
    "Customers may request a refund within 30 days of purchase. "
    "Digital downloads are non-refundable once accessed. "
    "Shipping fees are never refunded."
)


@pytest.fixture
def contexts(scored_factory):
    return [
        scored_factory("c1", REFUNDS, score=0.9, rank=1),
        scored_factory("c2", "Approved refunds settle within 5 to 10 business days.", 0.7, 2),
        scored_factory("c3", "Gift cards cannot be exchanged for cash.", 0.5, 3),
    ]


# --------------------------------------------------------------------------- #
# extract_markers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("A single cite [1].", [1]),
        ("Comma group [1,2].", [1, 2]),
        ("Spaced group [1, 2].", [1, 2]),
        ("Adjacent groups [1][2].", [1, 2]),
        ("Padded [ 3 ] marker.", [3]),
        ("Multi digit [12] marker.", [12]),
        ("Zero [0] marker.", [0]),
        ("Mixed [3][1, 3] repeats.", [3, 1]),
    ],
)
def test_extract_markers_accepts_numeric_groups(text, expected):
    assert extract_markers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "the array [i] is indexed",
        "a bracket [not a number] in prose",
        "an empty pair [] of brackets",
        "whitespace only [ ] here",
        "trailing letters [1a] do not count",
        "no brackets at all",
        "",
    ],
)
def test_extract_markers_rejects_non_markers(text):
    assert extract_markers(text) == []


def test_extract_markers_is_ordered_by_first_appearance_and_deduplicated():
    assert extract_markers("B [4] then A [2] then B again [4].") == [4, 2]


# --------------------------------------------------------------------------- #
# resolve_citations
# --------------------------------------------------------------------------- #
def test_resolve_citations_maps_markers_positionally(contexts):
    citations, invalid = resolve_citations("First [1]. Third [3].", contexts)

    assert invalid == []
    assert [c.marker for c in citations] == [1, 3]
    assert [c.chunk_id for c in citations] == ["c1", "c3"]
    assert citations[0].score == pytest.approx(0.9)
    assert citations[1].title == "Doc"
    assert citations[1].page is None


def test_resolve_citations_flags_out_of_range_and_zero(contexts):
    citations, invalid = resolve_citations("Bad [0] worse [4] worst [99].", contexts)

    assert citations == []
    assert invalid == [0, 4, 99]


def test_resolve_citations_mixes_valid_and_invalid(contexts):
    citations, invalid = resolve_citations("Good [2] bad [7].", contexts)

    assert [c.marker for c in citations] == [2]
    assert invalid == [7]


def test_resolve_citations_on_an_answer_citing_nothing(contexts):
    citations, invalid = resolve_citations("No sources were used here.", contexts)

    assert (citations, invalid) == ([], [])


def test_resolve_citations_quotes_the_sentence_matching_the_claim(contexts):
    citations, _ = resolve_citations("Shipping fees are never refunded [1].", contexts)
    assert citations[0].quote == "Shipping fees are never refunded."

    citations, _ = resolve_citations(
        "Digital downloads cannot be refunded once accessed [1].", contexts
    )
    assert citations[0].quote == "Digital downloads are non-refundable once accessed."


def test_resolve_citations_quote_falls_back_to_the_first_sentence(contexts):
    """A marker with no surrounding prose has nothing to match against."""
    citations, _ = resolve_citations("[1]", contexts)
    assert citations[0].quote == "Customers may request a refund within 30 days of purchase."


def test_resolve_citations_truncates_long_quotes(scored_factory):
    long_sentence = "Refund eligibility depends on " + "several documented factors " * 20
    citations, _ = resolve_citations(
        "Refund eligibility depends on documented factors [1].",
        [scored_factory("c1", long_sentence)],
    )

    quote = citations[0].quote
    assert len(quote) <= QUOTE_MAX_CHARS
    assert quote.endswith("...")


def test_resolve_citations_is_deterministic(contexts):
    answer = "Refunds settle fast [2] and shipping is excluded [1]."
    first = resolve_citations(answer, contexts)
    second = resolve_citations(answer, contexts)

    assert [c.model_dump() for c in first[0]] == [c.model_dump() for c in second[0]]
    assert first[1] == second[1]


# --------------------------------------------------------------------------- #
# renumber_answer
# --------------------------------------------------------------------------- #
def test_renumber_answer_leaves_an_uncited_answer_untouched(contexts):
    answer = "I could not find enough support for that in the indexed documents."
    assert renumber_answer(answer, contexts) == (answer, [], [])


def test_renumber_answer_strips_an_answer_citing_only_invalid_markers(contexts):
    rewritten, citations, invalid = renumber_answer(
        "Refunds take 30 days [9]. Digital goods are final [4].", contexts
    )

    assert rewritten == "Refunds take 30 days. Digital goods are final."
    assert citations == []
    assert invalid == [9, 4]


def test_renumber_answer_densifies_survivors(contexts):
    rewritten, citations, invalid = renumber_answer(
        "Settlement is quick [2]. Shipping is excluded [1].", contexts
    )

    assert rewritten == "Settlement is quick [1]. Shipping is excluded [2]."
    assert [c.marker for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == ["c2", "c1"]
    assert invalid == []


def test_renumber_answer_mixes_valid_and_invalid(contexts):
    rewritten, citations, invalid = renumber_answer(
        "Refunds settle quickly [2] and gift cards are excluded [8].", contexts
    )

    assert rewritten == "Refunds settle quickly [1] and gift cards are excluded."
    assert [c.marker for c in citations] == [1]
    assert citations[0].chunk_id == "c2"
    assert invalid == [8]


def test_renumber_answer_collapses_adjacent_groups(contexts):
    rewritten, citations, invalid = renumber_answer(
        "Refunds settle quickly [3][7].", contexts
    )

    assert rewritten == "Refunds settle quickly [1]."
    assert [c.chunk_id for c in citations] == ["c3"]
    assert invalid == [7]


def test_renumber_answer_keeps_a_surviving_pair_of_adjacent_groups(contexts):
    rewritten, citations, _ = renumber_answer("Both apply [3][1].", contexts)

    assert rewritten == "Both apply [1][2]."
    assert [c.chunk_id for c in citations] == ["c3", "c1"]


def test_renumber_answer_rewrites_comma_groups(contexts):
    rewritten, citations, invalid = renumber_answer("Two sources agree [2, 3].", contexts)

    assert rewritten == "Two sources agree [1, 2]."
    assert [c.marker for c in citations] == [1, 2]
    assert invalid == []


def test_renumber_answer_prunes_a_comma_group(contexts):
    rewritten, citations, invalid = renumber_answer("Partly sourced [2,9].", contexts)

    assert rewritten == "Partly sourced [1]."
    assert [c.chunk_id for c in citations] == ["c2"]
    assert invalid == [9]


def test_renumber_answer_leaves_no_whitespace_scars(contexts):
    rewritten, _, invalid = renumber_answer(
        "[9] Refunds are slow [5] but shipping is excluded [1] , mostly [6].", contexts
    )

    assert rewritten == "Refunds are slow but shipping is excluded [1], mostly."
    assert invalid == [9, 5, 6]
    assert "  " not in rewritten
    assert " ." not in rewritten


def test_renumber_answer_is_byte_identical_when_already_dense(contexts):
    answer = "Refunds settle quickly [1].  Shipping is excluded [2]."
    rewritten, citations, invalid = renumber_answer(answer, contexts)

    assert rewritten == answer
    assert [c.marker for c in citations] == [1, 2]
    assert invalid == []


def test_renumber_answer_ignores_brackets_in_ordinary_prose(contexts):
    rewritten, citations, invalid = renumber_answer(
        "The array [i] is indexed, per the guide [2], unlike [not a number].", contexts
    )

    assert rewritten == "The array [i] is indexed, per the guide [1], unlike [not a number]."
    assert [c.chunk_id for c in citations] == ["c2"]
    assert invalid == []


def test_renumber_answer_output_stays_consistent_with_its_citations(contexts):
    rewritten, citations, invalid = renumber_answer(
        "A [3] and B [9] and C [1,3] and D [2].", contexts
    )

    # Every marker left in the prose has a citation, and the list is dense.
    assert extract_markers(rewritten) == [c.marker for c in citations]
    assert [c.marker for c in citations] == list(range(1, len(citations) + 1))
    assert not set(extract_markers(rewritten)) & set(invalid)
    assert invalid == [9]


def test_renumber_answer_quotes_are_chosen_against_the_original_prose(contexts):
    _, citations, _ = renumber_answer(
        "Bogus [7]. Shipping fees are never refunded [1].", contexts
    )

    assert citations[0].marker == 1
    assert citations[0].quote == "Shipping fees are never refunded."


def test_renumber_answer_with_no_contexts_drops_every_marker():
    rewritten, citations, invalid = renumber_answer("Sourced [1] and [2].", [])

    assert rewritten == "Sourced and."
    assert citations == []
    assert invalid == [1, 2]


# --------------------------------------------------------------------------- #
# strip_markers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Refunds take 30 days [1].", "Refunds take 30 days."),
        ("Two sources [1, 2] agree.", "Two sources agree."),
        ("Adjacent [1][2] groups.", "Adjacent groups."),
        ("[1] Leading marker.", "Leading marker."),
        ("The array [i] is fine.", "The array [i] is fine."),
        ("Nothing to strip.", "Nothing to strip."),
        ("", ""),
    ],
)
def test_strip_markers(text, expected):
    assert strip_markers(text) == expected


def test_strip_markers_preserves_paragraph_breaks():
    assert strip_markers("First [1].\n\nSecond [2].") == "First.\n\nSecond."


def test_unicode_bracket_variants_are_read_as_markers():
    """Not every model cites with ASCII brackets.

    gpt-oss emits CJK lenticular brackets. The citation is well formed; only the
    codepoint differs, and reading it as "cited nothing" would score a correct,
    properly attributed answer as ungrounded.
    """
    assert extract_markers("acknowledged within 5 minutes\u30101\u3011.") == [1]
    assert extract_markers("see \uff3b2\uff3d and \u27e63\u27e7") == [2, 3]
    # ASCII must keep working, and prose brackets must still be ignored.
    assert extract_markers("a [1] and [2, 3] here") == [1, 2, 3]
    assert extract_markers("the [i] of prose \u3010not a number\u3011") == []


def test_strip_markers_removes_unicode_variants():
    assert strip_markers("within 5 minutes\u30101\u3011.") == "within 5 minutes."
