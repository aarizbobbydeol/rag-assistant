"""Offline evaluation of one pipeline configuration against the golden set.

Run it as ``python -m eval.run_eval`` (see ``--help``). Every number the README
quotes comes out of this module, and the ablation sweep in ``eval.sweep`` is
just this function called in a loop with different settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import RetrievalMode, SplitterName
from app.pipeline import RagPipeline
from eval.dataset import EvalExample, load_dataset, resolve_relevant_chunks
from eval.metrics import (
    aggregate,
    citation_precision,
    citation_recall,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = PROJECT_ROOT / "data" / "golden" / "qa.jsonl"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus"


@dataclass
class ExampleResult:
    question: str
    tags: list[str]
    must_abstain: bool
    abstained: bool
    n_relevant: int
    retrieval: dict[str, float]
    answer: dict[str, float]
    latency_ms: float


@dataclass
class EvalReport:
    config: dict[str, Any]
    n_examples: int
    retrieval: dict[str, float]
    answer: dict[str, float]
    abstention: dict[str, float]
    by_tag: dict[str, dict[str, float]]
    wall_clock_s: float
    per_example: list[ExampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["per_example"] = [asdict(r) for r in self.per_example]
        return payload


def build_settings(
    base: Settings | None = None,
    *,
    index_dir: Path | None = None,
    **overrides: Any,
) -> Settings:
    """A Settings clone with overrides applied and persistence sandboxed.

    Each evaluation run gets its own index directory; without that, a sweep
    would silently reuse the previous configuration's vectors.
    """
    data = (base or Settings()).model_dump()
    data.update(overrides)
    scratch = index_dir or Path(tempfile.mkdtemp(prefix="rag-eval-"))
    data["index_dir"] = scratch
    data["upload_dir"] = scratch / "uploads"
    data["persist_index"] = False
    return Settings(**data)


def evaluate(
    settings: Settings,
    examples: Sequence[EvalExample],
    corpus_dir: Path = DEFAULT_CORPUS,
    top_k: int | None = None,
    pipeline: RagPipeline | None = None,
    generate: bool = True,
) -> EvalReport:
    started = time.perf_counter()
    top_k = top_k or settings.top_k

    if pipeline is None:
        pipeline = RagPipeline(settings)
        pipeline.index.clear()
        ingested = pipeline.ingest_paths([corpus_dir], recursive=True)
        if ingested.total_chunks == 0:
            raise SystemExit(
                f"No documents indexed from {corpus_dir}. "
                "Run `python scripts/make_sample_corpus.py` first."
            )

    chunks = list(pipeline.index.chunks.values())
    results: list[ExampleResult] = []

    for example in examples:
        relevant = resolve_relevant_chunks(example, chunks)
        q_started = time.perf_counter()
        contexts = pipeline.index.search(
            example.question, top_k, settings.retrieval_mode, settings.rerank_enabled
        )
        ranked = [c.chunk.chunk_id for c in contexts]

        retrieval = {
            "recall@k": recall_at_k(ranked, relevant, top_k),
            "precision@k": precision_at_k(ranked, relevant, top_k),
            "hit_rate@k": hit_rate_at_k(ranked, relevant, top_k),
            "mrr": mrr(ranked, relevant),
            "ndcg@k": ndcg_at_k(ranked, relevant, top_k),
        }

        answer_metrics: dict[str, float] = {}
        abstained = False
        if generate:
            result = pipeline.answer(
                example.question, top_k=top_k, mode=settings.retrieval_mode,
                rerank=settings.rerank_enabled, include_contexts=False,
            )
            abstained = result.groundedness.abstained
            cited = [c.chunk_id for c in result.citations]
            answer_metrics = {
                "citation_precision": citation_precision(cited, relevant),
                "citation_recall": citation_recall(cited, relevant),
                "groundedness": result.groundedness.score,
                "token_f1": 0.0 if example.must_abstain else token_f1(result.answer, example.answer),
                "rouge_l": 0.0 if example.must_abstain else rouge_l(result.answer, example.answer),
                "n_citations": float(len(result.citations)),
            }

        results.append(
            ExampleResult(
                question=example.question,
                tags=list(example.tags),
                must_abstain=example.must_abstain,
                abstained=abstained,
                n_relevant=len(relevant),
                retrieval=retrieval,
                answer=answer_metrics,
                latency_ms=round((time.perf_counter() - q_started) * 1000, 2),
            )
        )

    answerable = [r for r in results if not r.must_abstain]
    unanswerable = [r for r in results if r.must_abstain]

    report = EvalReport(
        config=_config_summary(settings, top_k),
        n_examples=len(results),
        retrieval=aggregate([r.retrieval for r in answerable]),
        answer=aggregate([r.answer for r in answerable if r.answer]),
        abstention=_abstention_metrics(answerable, unanswerable),
        by_tag=_by_tag(results),
        wall_clock_s=round(time.perf_counter() - started, 2),
        per_example=results,
    )
    return report


def _config_summary(settings: Settings, top_k: int) -> dict[str, Any]:
    return {
        "splitter": settings.splitter.value,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "mode": settings.retrieval_mode.value,
        "rerank": settings.rerank_enabled,
        "rerank_backend": settings.rerank_backend,
        "top_k": top_k,
        "candidate_k": settings.candidate_k,
        "embedding_backend": settings.embedding_backend,
        "llm_backend": settings.llm_backend,
    }


def _abstention_metrics(answerable: list[ExampleResult], unanswerable: list[ExampleResult]) -> dict[str, float]:
    """How reliably the assistant refuses when the corpus has no answer.

    ``false_answer_rate`` is the one that matters for a hallucination guard: the
    fraction of genuinely unanswerable questions the system answered anyway.
    """
    correct_refusals = sum(1 for r in unanswerable if r.abstained)
    wrong_refusals = sum(1 for r in answerable if r.abstained)
    return {
        "n_unanswerable": float(len(unanswerable)),
        "abstain_recall": correct_refusals / len(unanswerable) if unanswerable else 0.0,
        "false_answer_rate": (
            (len(unanswerable) - correct_refusals) / len(unanswerable) if unanswerable else 0.0
        ),
        "over_refusal_rate": wrong_refusals / len(answerable) if answerable else 0.0,
    }


def _by_tag(results: list[ExampleResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[ExampleResult]] = {}
    for row in results:
        for tag in row.tags or ["untagged"]:
            buckets.setdefault(tag, []).append(row)
    return {
        tag: {
            "n": float(len(rows)),
            **aggregate([r.retrieval for r in rows]),
        }
        for tag, rows in sorted(buckets.items())
    }


def format_report(report: EvalReport) -> str:
    lines = [
        "",
        "=" * 72,
        f"  {report.n_examples} examples   |   {report.wall_clock_s}s wall clock",
        "=" * 72,
        "  config: " + ", ".join(f"{k}={v}" for k, v in report.config.items()),
        "",
        "  RETRIEVAL (answerable questions only)",
    ]
    for key, value in sorted(report.retrieval.items()):
        if key.endswith("_stdev") or key == "n":
            continue
        lines.append(f"    {key:<24} {value:.3f}")
    if report.answer:
        lines.append("")
        lines.append("  ANSWER")
        for key, value in sorted(report.answer.items()):
            if key.endswith("_stdev") or key == "n":
                continue
            lines.append(f"    {key:<24} {value:.3f}")
    lines.append("")
    lines.append("  ABSTENTION")
    for key, value in sorted(report.abstention.items()):
        lines.append(f"    {key:<24} {value:.3f}")
    lines.append("")
    lines.append("  BY TAG (recall@k)")
    for tag, values in report.by_tag.items():
        lines.append(f"    {tag:<24} n={int(values['n']):<4} recall@k={values.get('recall@k', 0.0):.3f}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one RAG configuration.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--splitter", choices=[s.value for s in SplitterName], default=None)
    parser.add_argument("--mode", choices=[m.value for m in RetrievalMode], default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-generate", action="store_true", help="Retrieval metrics only (fast).")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {"top_k": args.top_k}
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.splitter:
        overrides["splitter"] = SplitterName(args.splitter)
    if args.mode:
        overrides["retrieval_mode"] = RetrievalMode(args.mode)
    if args.no_rerank:
        overrides["rerank_enabled"] = False

    settings = build_settings(**overrides)
    examples = load_dataset(args.golden)
    if args.limit:
        examples = examples[: args.limit]

    report = evaluate(
        settings, examples, corpus_dir=args.corpus, top_k=args.top_k, generate=not args.no_generate
    )
    print(format_report(report))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
