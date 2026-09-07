"""End-to-end demo: index the sample corpus and answer a few questions.

    python scripts/smoke.py
    python scripts/smoke.py --question "How long do refunds take?"

Runs entirely offline on the built-in embedder and extractive answerer unless
``RAG_LLM_API_KEY`` is set, which makes it a useful five-second check that the
whole pipeline is wired correctly.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.models import AnswerResult  # noqa: E402
from app.observability import configure_logging  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402

DEFAULT_QUESTIONS = [
    "How long is the on-call rotation?",
    "How quickly must the primary on-call engineer acknowledge a SEV-1 page?",
    "How many days of paid vacation do employees receive?",
    # Deliberately not in the corpus: this one should be refused, not answered.
    "What is the company's policy on cryptocurrency payments?",
]


def render(result: AnswerResult) -> str:
    lines: list[str] = []
    lines.append(f"\nQ: {result.question}")
    if result.standalone_question != result.question:
        lines.append(f"   (condensed: {result.standalone_question})")
    lines.append("")
    lines.append(textwrap.indent(textwrap.fill(result.answer, 90), "   "))
    lines.append("")

    if result.citations:
        lines.append("   Sources")
        for citation in result.citations:
            page = f" p.{citation.page}" if citation.page is not None else ""
            lines.append(f"     [{citation.marker}] {citation.title}{page}  ({citation.source})")
            if citation.quote:
                lines.append(textwrap.indent(textwrap.fill(f'"{citation.quote}"', 80), "         "))
    else:
        lines.append("   Sources: none (abstained)")

    grounded = result.groundedness
    lines.append(
        f"\n   groundedness={grounded.score:.2f} "
        f"({grounded.supported_sentences}/{grounded.total_sentences} sentences supported)"
        f"{'  ABSTAINED' if grounded.abstained else ''}"
    )
    if grounded.unsupported:
        lines.append(f"   unsupported: {grounded.unsupported[0][:80]}")
    latency = " ".join(f"{k}={v:.0f}ms" for k, v in result.latency_ms.items())
    lines.append(f"   {latency}   tokens={result.usage.total_tokens}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the RAG pipeline.")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    parser.add_argument("--question", action="append", dest="questions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="Suppress structured logs.")
    args = parser.parse_args(argv)

    configure_logging("WARNING" if args.quiet else "INFO", json_logs=False)

    settings = Settings(persist_index=False, top_k=args.top_k)
    pipeline = RagPipeline(settings)
    pipeline.index.clear()

    ingested = pipeline.ingest_paths([args.corpus], recursive=True)
    print(
        f"\nindexed {ingested.total_documents} documents "
        f"-> {ingested.total_chunks} chunks in {ingested.elapsed_ms:.0f}ms"
    )
    if ingested.skipped:
        print("skipped: " + "; ".join(ingested.skipped))
    if ingested.total_chunks == 0:
        print("\nNothing to search. Run `python scripts/make_sample_corpus.py` first.")
        return 1

    stats = pipeline.index.stats()
    print(
        f"embedder={stats['embedder']} dim={stats['embedding_dim']} "
        f"store={stats['vector_store']} reranker={stats['reranker']} "
        f"llm={pipeline.llm.provider}:{pipeline.llm.model}"
    )

    session_id = "smoke-session"
    for question in args.questions or DEFAULT_QUESTIONS:
        print(render(pipeline.answer(question, session_id=session_id)))

    print("\nconversation memory check")
    print(render(pipeline.answer("And what about SEV-2?", session_id=session_id)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
