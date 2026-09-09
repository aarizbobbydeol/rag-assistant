"""Tests for the document loading layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import DocumentLoadError, UnsupportedFileType
from app.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    build_page_spans,
    iter_paths,
    load_bytes,
    load_path,
)
from app.utils import normalize_whitespace, stable_id


# --------------------------------------------------------------------------- #
# A hand-assembled PDF, so the PDF tests need no writer dependency.
# --------------------------------------------------------------------------- #
def make_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """Build an uncompressed PDF whose text ``pypdf`` can extract.

    Objects are emitted in the order they are created and the xref table is
    built from the real byte offsets, so the result is a structurally valid
    file rather than something that only survives pypdf's recovery path.
    Lines must be plain ASCII without ``(``, ``)`` or ``\\``.
    """
    bodies: list[bytes] = []

    def add(body: bytes) -> int:
        bodies.append(body)
        return len(bodies)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = add(b"")  # reserved; the page tree needs the kid ids first
    kid_ids: list[int] = []
    for lines in pages:
        drawn = "".join(f"({line}) Tj T*\n" for line in lines)
        stream = f"BT /F1 14 Tf 18 TL 40 700 Td\n{drawn}ET\n".encode("ascii")
        content_id = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream"
        )
        kid_ids.append(
            add(
                b"<< /Type /Page /Parent "
                + str(pages_id).encode()
                + b" 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
                + str(font_id).encode()
                + b" 0 R >> >> /Contents "
                + str(content_id).encode()
                + b" 0 R >>"
            )
        )
    kids = b" ".join(str(kid).encode() + b" 0 R" for kid in kid_ids)
    bodies[pages_id - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(kid_ids)).encode() + b" >>"
    )
    catalog_id = add(b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>")
    info_id = add(b"<< /Title (" + (title or "").encode("ascii") + b") >>") if title else None

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    startxref = len(out)
    out += b"xref\n0 " + str(len(bodies) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = b"trailer\n<< /Size " + str(len(bodies) + 1).encode()
    trailer += b" /Root " + str(catalog_id).encode() + b" 0 R"
    if info_id is not None:
        trailer += b" /Info " + str(info_id).encode() + b" 0 R"
    out += trailer + b" >>\nstartxref\n" + str(startxref).encode() + b"\n%%EOF\n"
    return bytes(out)


def pages_covering(spans: list[list[int]], offset: int) -> list[int]:
    return [page for start, end, page in spans if start <= offset < end]


# --------------------------------------------------------------------------- #
# Plain text and markdown
# --------------------------------------------------------------------------- #
def test_markdown_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "refunds.md"
    path.write_text(
        "# Refund policy\n\n"
        "Customers may   request a refund within 30 days.\t\n\n\n\n"
        "## Exceptions\n\n"
        "Digital downloads are non-refundable.\n",
        encoding="utf-8",
    )

    doc = load_path(path)

    assert doc.source == str(path.resolve())
    assert doc.doc_id == stable_id(doc.source)
    assert doc.title == "Refund policy"
    assert "Customers may request a refund within 30 days." in doc.text
    assert "## Exceptions" in doc.text
    # normalize_whitespace collapsed the runs and trimmed the tail.
    assert normalize_whitespace(doc.text) == doc.text
    assert "\n\n\n" not in doc.text
    assert doc.metadata["suffix"] == ".md"
    assert doc.metadata["bytes"] == len(path.read_bytes())


def test_load_path_is_stable_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nAlpha beta gamma.\n", encoding="utf-8")
    assert load_path(path).model_dump() == load_path(str(path)).model_dump()


def test_title_prefers_h1_then_falls_back_to_stem(tmp_path: Path) -> None:
    with_heading = tmp_path / "release-notes.txt"
    with_heading.write_text("# Version 2.1\n\nAdds hybrid retrieval.\n", encoding="utf-8")
    assert load_path(with_heading).title == "Version 2.1"

    without_heading = tmp_path / "server_2024.log"
    without_heading.write_text("WARN cache miss\nINFO ready\n", encoding="utf-8")
    assert load_path(without_heading).title == "server_2024"


def test_lossy_decode_survives_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "mixed.txt"
    path.write_bytes(b"caf\xe9 latte and espresso")
    assert "latte and espresso" in load_path(path).text


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")

    with pytest.raises(UnsupportedFileType):
        load_path(path)
    with pytest.raises(UnsupportedFileType):
        load_bytes(b"anything", "archive.zip")
    with pytest.raises(UnsupportedFileType):
        load_bytes(b"anything", "no-suffix")
    assert ".zip" not in SUPPORTED_SUFFIXES


def test_empty_file_raises_document_load_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_path(empty)

    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\t\n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_path(blank)


def test_missing_path_raises_document_load_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_path(tmp_path / "nope.md")
    with pytest.raises(DocumentLoadError):
        iter_paths(tmp_path / "nowhere")


def test_malformed_json_raises_document_load_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_path(path)


# --------------------------------------------------------------------------- #
# Directory walking
# --------------------------------------------------------------------------- #
@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "sub" / "nested").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "notes.xyz").write_text("skip me\n", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# hidden\n", encoding="utf-8")
    (tmp_path / ".git" / "config.md").write_text("# git\n", encoding="utf-8")
    (tmp_path / "sub" / "z.md").write_text("# Z\n", encoding="utf-8")
    (tmp_path / "sub" / "nested" / "deep.md").write_text("# Deep\n", encoding="utf-8")
    return tmp_path


def test_iter_paths_is_ordered_and_filtered(corpus: Path) -> None:
    found = [p.relative_to(corpus).as_posix() for p in iter_paths(corpus)]
    assert found == ["a.txt", "b.md", "sub/nested/deep.md", "sub/z.md"]


def test_iter_paths_is_deterministic(corpus: Path) -> None:
    assert list(iter_paths(corpus)) == list(iter_paths(corpus))


def test_iter_paths_honours_recursive_flag(corpus: Path) -> None:
    found = [p.name for p in iter_paths(corpus, recursive=False)]
    assert found == ["a.txt", "b.md"]


def test_iter_paths_on_a_single_file(corpus: Path) -> None:
    assert list(iter_paths(corpus / "b.md")) == [corpus / "b.md"]
    assert list(iter_paths(corpus / "notes.xyz")) == []


# --------------------------------------------------------------------------- #
# HTML, JSON, CSV
# --------------------------------------------------------------------------- #
HTML_SAMPLE = """<!doctype html>
<html><head><title>Quarterly &amp; Report</title>
<style>body { color: crimson; }</style>
<script>var marker = "<b>script body</b>"; alert(1);</script>
</head>
<body>
<h1>Revenue</h1>
<p>Revenue grew 12% in Q3 &amp; Q4.</p>
<!-- an internal comment -->
<div>Costs &lt;fell&gt; sharply.</div>
</body></html>
"""


def test_html_tags_and_script_bodies_are_stripped(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    path.write_text(HTML_SAMPLE, encoding="utf-8")

    doc = load_path(path)

    assert doc.title == "Quarterly & Report"
    assert "Revenue grew 12% in Q3 & Q4." in doc.text
    assert "Costs <fell> sharply." in doc.text
    assert "Revenue" in doc.text
    for leaked in ("crimson", "alert(1)", "script body", "internal comment", "<p>", "<div>"):
        assert leaked not in doc.text


def test_json_is_pretty_printed_with_sorted_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"zeta": 1, "alpha": {"delta": 4, "beta": 2}}', encoding="utf-8")

    text = load_path(path).text

    assert text.index('"alpha"') < text.index('"zeta"')
    assert text.index('"beta"') < text.index('"delta"')


def test_csv_rows_render_as_readable_lines(tmp_path: Path) -> None:
    path = tmp_path / "team.csv"
    path.write_text("name,role\nAda, engineer\n\nGrace,admiral\n", encoding="utf-8")

    doc = load_path(path)

    assert doc.text.splitlines() == ["name | role", "Ada | engineer", "Grace | admiral"]
    assert doc.metadata["row_count"] == 3
    assert doc.metadata["column_count"] == 2


def test_load_bytes_uses_the_filename_as_source() -> None:
    doc = load_bytes(b"# Uploaded\n\nBody text.\n", "uploads/manual.md")
    assert doc.source == "uploads/manual.md"
    assert doc.doc_id == stable_id("uploads/manual.md")
    assert doc.title == "Uploaded"


# --------------------------------------------------------------------------- #
# Page spans
# --------------------------------------------------------------------------- #
def test_build_page_spans_partitions_every_offset() -> None:
    pages = ["alpha", "", "gamma"]

    text, spans = build_page_spans(pages)

    assert text == "alpha\n\ngamma"
    assert spans == [[0, 6, 1], [6, 7, 2], [7, 12, 3]]
    assert spans[-1][1] == len(text)
    assert all(pages_covering(spans, offset) == [spans[i][2]]
               for i, (start, end, _page) in enumerate(spans)
               for offset in range(start, end))


def test_build_page_spans_handles_a_single_page() -> None:
    text, spans = build_page_spans(["only page"])
    assert text == "only page"
    assert spans == [[0, len(text), 1]]

    assert build_page_spans([]) == ("", [])


def test_pdf_page_spans_cover_the_text_exactly() -> None:
    data = make_pdf(
        [
            ["Alpha page one about widgets", "The first page continues here"],
            ["Beta page two about gadgets", "The second page continues here"],
        ],
        title="Widget Manual",
    )

    doc = load_bytes(data, "manual.pdf")
    spans = doc.metadata["page_spans"]

    assert doc.title == "Widget Manual"  # PDF metadata title wins over any heading
    assert doc.metadata["page_count"] == 2
    assert len(spans) == 2
    assert "widgets" in doc.text and "gadgets" in doc.text

    # No gaps, no overlaps, full coverage, 1-based pages.
    assert spans[0][0] == 0
    assert spans[-1][1] == len(doc.text)
    assert [page for _s, _e, page in spans] == [1, 2]
    for previous, following in zip(spans, spans[1:], strict=False):
        assert previous[1] == following[0]
    for offset in range(len(doc.text)):
        assert len(pages_covering(spans, offset)) == 1

    # Each span slices back to that page's own words.
    assert "widgets" in doc.text[spans[0][0]:spans[0][1]]
    assert "gadgets" in doc.text[spans[1][0]:spans[1][1]]
    assert "gadgets" not in doc.text[spans[0][0]:spans[0][1]]

    # Normalising happened per page, so the offsets cannot have drifted.
    assert normalize_whitespace(doc.text) == doc.text


def test_pdf_without_metadata_title_falls_back_to_stem(tmp_path: Path) -> None:
    path = tmp_path / "unnamed-report.pdf"
    path.write_bytes(make_pdf([["Only one line of text"]]))

    doc = load_path(path)

    assert doc.title == "unnamed-report"
    assert doc.metadata["page_spans"] == [[0, len(doc.text), 1]]


# --------------------------------------------------------------------------- #
# PDFs: wrapped lines and scans
# --------------------------------------------------------------------------- #
def test_pdf_wrapped_lines_are_rejoined_into_sentences():
    """A PDF breaks lines for layout, not for meaning.

    Left alone, a sentence that wrapped becomes two "sentences", and an
    extractive answer quotes "must maintain a temperature between" and stops -
    dropping the number, which was the entire answer.
    """
    from app.ingestion.loaders import _unwrap_pdf_lines

    wrapped = (
        "Refrigerated trailers must maintain a temperature between\n"
        "2 and 8 degrees Celsius at all times during transit.\n"
    )
    assert "between 2 and 8 degrees" in _unwrap_pdf_lines(wrapped)


def test_unwrap_keeps_real_structure():
    """Headings, list items and finished sentences own their line breaks."""
    from app.ingestion.loaders import _unwrap_pdf_lines

    structured = "# Severity levels\n- SEV-1 is a total outage.\n- SEV-2 is degradation.\nDone."
    assert _unwrap_pdf_lines(structured).count("\n") == 3


def test_unwrap_rejoins_a_hyphenated_word_without_a_space():
    from app.ingestion.loaders import _unwrap_pdf_lines

    assert "quarantined" in _unwrap_pdf_lines("the load must be quaran-\ntined immediately.")


def test_a_scanned_pdf_says_it_is_a_scan(monkeypatch):
    """"No extractable text" is true but useless when the page is full of words.

    The user needs to know the file is a scan and that OCR is what reads it.
    """
    import shutil as shutil_module

    from app.errors import DocumentLoadError
    from app.ingestion import loaders

    monkeypatch.setattr(loaders, "_pdf_pages_pypdf", lambda data: ([""], ""))
    monkeypatch.setattr(loaders, "_pdf_pages_pymupdf", lambda data, *, ocr: ([], ""))
    monkeypatch.setattr(shutil_module, "which", lambda name: None)

    with pytest.raises(DocumentLoadError) as caught:
        loaders._extract_pdf(b"%PDF-1.4 fake")

    assert "scan" in caught.value.message.lower()
    assert "ocr" in (caught.value.detail or "").lower()


def test_ocr_is_only_attempted_when_the_text_layer_is_empty(monkeypatch):
    """OCR is slow, so a normal PDF must never pay for it."""
    from app.ingestion import loaders

    calls: list[bool] = []

    def fake_pymupdf(data, *, ocr):
        calls.append(ocr)
        return [], ""

    monkeypatch.setattr(loaders, "_pdf_pages_pypdf", lambda data: (["Real text here."], "T"))
    monkeypatch.setattr(loaders, "_pdf_pages_pymupdf", fake_pymupdf)

    text, _, extra = loaders._extract_pdf(b"%PDF-1.4 fake")
    assert "Real text here." in text
    assert extra["extractor"] == "pypdf"
    assert calls == []
