"""Turn files and byte payloads into :class:`~app.models.Document` objects.

Extraction is deliberately dependency-light: PDFs go through ``pypdf`` (a
declared dependency) and everything else is handled with the standard library,
so the ingestion layer works in an offline container with no parser stack.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.errors import DocumentLoadError, UnsupportedFileType
from app.models import Document
from app.observability import get_logger
from app.utils import normalize_whitespace, stable_id

logger = get_logger(__name__)

SUPPORTED_SUFFIXES: set[str] = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".rst",
    ".log",
}

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.IGNORECASE | re.DOTALL
)
_HTML_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|br|hr|li|ul|ol|dl|dt|dd|tr|td|th|table|thead|tbody|section|article"
    r"|header|footer|nav|aside|main|form|figure|figcaption|blockquote|pre|h[1-6])\b[^>]*/?>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_H1_RE = re.compile(r"^#\s+(\S.*?)\s*$", re.MULTILINE)


def build_page_spans(pages: list[str]) -> tuple[str, list[list[int]]]:
    """Concatenate ``pages`` with a single newline and tile them with spans.

    Returns ``(text, spans)`` where each span is ``[start_char, end_char, page_no]``
    with 1-based page numbers. The separator newline is charged to the page it
    follows, so the spans partition ``range(len(text))`` exactly: no gaps, no
    overlaps, and every character offset resolves to exactly one page. Pages
    must already be normalised - normalising the joined text afterwards would
    shift every offset.
    """
    last = len(pages) - 1
    spans: list[list[int]] = []
    cursor = 0
    for index, page in enumerate(pages):
        end = cursor + len(page) + (1 if index < last else 0)
        spans.append([cursor, end, index + 1])
        cursor = end
    return "\n".join(pages), spans


def load_path(path: str | Path) -> Document:
    """Read one file from disk.

    ``source`` is the resolved absolute path and ``doc_id`` is its
    :func:`~app.utils.stable_id`, so re-ingesting the same file replaces rather
    than duplicates it.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise DocumentLoadError("File not found", str(file_path))
    if not file_path.is_file():
        raise DocumentLoadError("Not a regular file", str(file_path))
    _check_suffix(file_path.name)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError("Could not read file", f"{file_path}: {exc}") from exc
    return _build(data, file_path.name, str(file_path.resolve()))


def load_bytes(data: bytes, filename: str) -> Document:
    """Load an in-memory payload (an upload) using ``filename`` for the format.

    The contract fixes ``doc_id = stable_id(resolved_source_path)``; a byte
    payload has no path on disk, so ``filename`` *is* its source and the id is
    derived from it verbatim. That keeps ``doc_id == stable_id(doc.source)``
    true for both loaders, which is what callers dedupe on.
    """
    _check_suffix(filename)
    return _build(data, Path(filename).name, filename)


def iter_paths(root: str | Path, recursive: bool = True) -> Iterator[Path]:
    """Yield loadable files under ``root`` in a deterministic order.

    Dotfiles and dot-directories are skipped (``.git``, ``.venv``, editor
    droppings), as are suffixes outside :data:`SUPPORTED_SUFFIXES`. Passing a
    file yields just that file when it is loadable. The walk is materialised
    eagerly so a bad ``root`` fails at the call site rather than mid-iteration.
    """
    base = Path(root).expanduser()
    if not base.exists():
        raise DocumentLoadError("Path not found", str(base))
    if base.is_file():
        loadable = not base.name.startswith(".") and base.suffix.lower() in SUPPORTED_SUFFIXES
        return iter([base] if loadable else [])
    return iter(list(_walk(base, recursive)))


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _walk(directory: Path, recursive: bool) -> Iterator[Path]:
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if recursive:
                yield from _walk(entry, True)
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_SUFFIXES:
            yield entry


def _check_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(
            f"Unsupported file type: {suffix or '(none)'}",
            f"{filename}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )
    return suffix


def _build(data: bytes, filename: str, source: str) -> Document:
    suffix = _check_suffix(filename)
    if suffix == ".pdf":
        text, title_hint, extra = _extract_pdf(data)
    elif suffix in (".html", ".htm"):
        text, title_hint, extra = _extract_html(data)
    elif suffix == ".json":
        text, title_hint, extra = _extract_json(data)
    elif suffix == ".csv":
        text, title_hint, extra = _extract_csv(data)
    else:
        text, title_hint, extra = _extract_plain(data)

    if not text.strip():
        raise DocumentLoadError("No extractable text", source)

    metadata: dict[str, Any] = {"suffix": suffix, "bytes": len(data), **extra}
    document = Document(
        doc_id=stable_id(source),
        source=source,
        title=title_hint or _first_h1(text) or Path(filename).stem or source,
        text=text,
        metadata=metadata,
    )
    logger.debug(
        "loaded document",
        extra={"source": source, "suffix": suffix, "characters": len(text)},
    )
    return document


def _first_h1(text: str) -> str:
    match = _H1_RE.search(text)
    return match.group(1).strip() if match else ""


def _decode(data: bytes) -> str:
    """Lossy utf-8 decode: a stray byte should cost one glyph, not the file."""
    return data.decode("utf-8", errors="replace")


def _extract_plain(data: bytes) -> tuple[str, str, dict[str, Any]]:
    return normalize_whitespace(_decode(data)), "", {}


def _extract_pdf(data: bytes) -> tuple[str, str, dict[str, Any]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many "protected" PDFs open with the empty owner password.
            reader.decrypt("")
        pages = [normalize_whitespace(page.extract_text() or "") for page in reader.pages]
        raw_title = getattr(reader.metadata, "title", None) or ""
    except Exception as exc:  # pypdf raises a wide family of parse errors
        raise DocumentLoadError("Could not parse PDF", str(exc)) from exc

    text, spans = build_page_spans(pages)
    return text, normalize_whitespace(str(raw_title)), {
        "page_count": len(pages),
        "page_spans": spans,
    }


def _extract_html(data: bytes) -> tuple[str, str, dict[str, Any]]:
    """Strip markup with regexes rather than pulling in an HTML parser.

    Good enough for ingestion: script/style bodies and the ``<title>`` element
    are dropped whole, block-level tags become newlines so paragraphs survive,
    and entities are unescaped only once every tag is gone - so escaped markup
    quoted in the prose is never re-interpreted as markup.
    """
    raw = _COMMENT_RE.sub(" ", _decode(data))
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)

    title = ""
    title_match = _HTML_TITLE_RE.search(raw)
    if title_match:
        title = normalize_whitespace(html.unescape(_TAG_RE.sub("", title_match.group(1))))
        raw = raw[: title_match.start()] + " " + raw[title_match.end() :]

    raw = _BLOCK_TAG_RE.sub("\n", raw)
    raw = _TAG_RE.sub(" ", raw)
    return normalize_whitespace(html.unescape(raw)), title, {}


def _extract_json(data: bytes) -> tuple[str, str, dict[str, Any]]:
    try:
        payload = json.loads(_decode(data))
    except ValueError as exc:
        raise DocumentLoadError("Invalid JSON", str(exc)) from exc
    # sort_keys keeps the rendered text - and every chunk id derived from it -
    # identical across runs and across key insertion orders.
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return normalize_whitespace(rendered), "", {}


def _extract_csv(data: bytes) -> tuple[str, str, dict[str, Any]]:
    try:
        rows = list(csv.reader(io.StringIO(_decode(data), newline="")))
    except csv.Error as exc:
        raise DocumentLoadError("Invalid CSV", str(exc)) from exc
    lines = [
        " | ".join(cell.strip() for cell in row)
        for row in rows
        if any(cell.strip() for cell in row)
    ]
    return normalize_whitespace("\n".join(lines)), "", {
        "row_count": len(lines),
        "column_count": max((len(row) for row in rows), default=0),
    }
