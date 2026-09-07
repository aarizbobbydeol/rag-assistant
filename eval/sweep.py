"""Ablation sweep over chunking and retrieval settings.

This is the part of the project that turns "I tuned the retriever" into a table
somebody can check. It re-indexes the corpus for every chunking configuration
(cheap for a demo corpus, and the only honest way to measure chunking) and
reuses that index across the retrieval variants that do not change chunking.

    python -m eval.sweep --quick
    python -m eval.sweep --out docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models import RetrievalMode, SplitterName
from app.pipeline import RagPipeline
from eval.dataset import load_dataset
from eval.run_eval import DEFAULT_CORPUS, DEFAULT_GOLDEN, build_settings, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ChunkConfig:
    splitter: SplitterName
    chunk_size: int
    chunk_overlap: int

    def label(self) -> str:
        return f"{self.splitter.value}/{self.chunk_size}/{self.chunk_overlap}"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode
    rerank: bool
    top_k: int

    def label(self) -> str:
        return f"{self.mode.value}{'+rerank' if self.rerank else ''}@{self.top_k}"


FULL_CHUNKS = [
    ChunkConfig(SplitterName.RECURSIVE, 400, 60),
    ChunkConfig(SplitterName.RECURSIVE, 700, 120),
    ChunkConfig(SplitterName.RECURSIVE, 1100, 180),
    ChunkConfig(SplitterName.FIXED, 700, 120),
    ChunkConfig(SplitterName.SENTENCE, 700, 120),
    ChunkConfig(SplitterName.SEMANTIC, 700, 120),
]
FULL_RETRIEVAL = [
    RetrievalConfig(RetrievalMode.DENSE, False, 5),
    RetrievalConfig(RetrievalMode.LEXICAL, False, 5),
    RetrievalConfig(RetrievalMode.HYBRID, False, 5),
    RetrievalConfig(RetrievalMode.HYBRID, True, 5),
    RetrievalConfig(RetrievalMode.HYBRID, True, 8),
]

QUICK_CHUNKS = [FULL_CHUNKS[1], FULL_CHUNKS[0]]
QUICK_RETRIEVAL = [FULL_RETRIEVAL[0], FULL_RETRIEVAL[1], FULL_RETRIEVAL[2], FULL_RETRIEVAL[3]]


def run_sweep(
    chunk_configs: Sequence[ChunkConfig],
    retrieval_configs: Sequence[RetrievalConfig],
    golden: Path = DEFAULT_GOLDEN,
    corpus: Path = DEFAULT_CORPUS,
    generate: bool = True,
) -> list[dict[str, Any]]:
    examples = load_dataset(golden)
    rows: list[dict[str, Any]] = []
    total = len(chunk_configs) * len(retrieval_configs)
    done = 0

    for chunk_cfg in chunk_configs:
        # Index once per chunking configuration, then vary retrieval on top of it.
        base = build_settings(
            splitter=chunk_cfg.splitter,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
        )
        pipeline = RagPipeline(base)
        pipeline.index.clear()
        ingested = pipeline.ingest_paths([corpus], recursive=True)
        if ingested.total_chunks == 0:
            raise SystemExit(f"No documents indexed from {corpus}.")

        for retrieval_cfg in retrieval_configs:
            done += 1
            started = time.perf_counter()
            settings = build_settings(
                base,
                index_dir=Path(base.index_dir),
                retrieval_mode=retrieval_cfg.mode,
                rerank_enabled=retrieval_cfg.rerank,
                top_k=retrieval_cfg.top_k,
            )
            pipeline.settings = settings
            pipeline.index.settings = settings
            report = evaluate(
                settings,
                examples,
                corpus_dir=corpus,
                top_k=retrieval_cfg.top_k,
                pipeline=pipeline,
                generate=generate,
            )
            rows.append(
                {
                    "chunking": chunk_cfg.label(),
                    "retrieval": retrieval_cfg.label(),
                    "chunks_indexed": ingested.total_chunks,
                    "recall@k": report.retrieval.get("recall@k", 0.0),
                    "ndcg@k": report.retrieval.get("ndcg@k", 0.0),
                    "mrr": report.retrieval.get("mrr", 0.0),
                    "hit_rate@k": report.retrieval.get("hit_rate@k", 0.0),
                    "citation_precision": report.answer.get("citation_precision", 0.0),
                    "groundedness": report.answer.get("groundedness", 0.0),
                    "false_answer_rate": report.abstention.get("false_answer_rate", 0.0),
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            print(
                f"[{done}/{total}] {chunk_cfg.label():<22} {retrieval_cfg.label():<18} "
                f"recall@k={rows[-1]['recall@k']:.3f} ndcg@k={rows[-1]['ndcg@k']:.3f} "
                f"({rows[-1]['seconds']}s)",
                flush=True,
            )

    return rows


def to_markdown(rows: list[dict[str, Any]], generate: bool) -> str:
    if not rows:
        return "_no results_\n"
    best = max(rows, key=lambda r: (r["ndcg@k"], r["recall@k"]))

    headers = ["chunking", "retrieval", "chunks", "recall@k", "nDCG@k", "MRR", "hit@k"]
    if generate:
        headers += ["cite-P", "grounded", "false-answer"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in sorted(rows, key=lambda r: -r["ndcg@k"]):
        marker = " **<-- best**" if row is best else ""
        cells = [
            row["chunking"],
            row["retrieval"],
            str(row["chunks_indexed"]),
            f"{row['recall@k']:.3f}",
            f"{row['ndcg@k']:.3f}",
            f"{row['mrr']:.3f}",
            f"{row['hit_rate@k']:.3f}",
        ]
        if generate:
            cells += [
                f"{row['citation_precision']:.3f}",
                f"{row['groundedness']:.3f}",
                f"{row['false_answer_rate']:.3f}",
            ]
        lines.append("| " + " | ".join(cells) + marker + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablate chunking and retrieval settings.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--quick", action="store_true", help="A small grid for a fast signal.")
    parser.add_argument("--no-generate", action="store_true", help="Retrieval metrics only.")
    parser.add_argument("--out", type=Path, default=None, help="Write a markdown table here.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    chunk_configs = QUICK_CHUNKS if args.quick else FULL_CHUNKS
    retrieval_configs = QUICK_RETRIEVAL if args.quick else FULL_RETRIEVAL

    rows = run_sweep(
        chunk_configs,
        retrieval_configs,
        golden=args.golden,
        corpus=args.corpus,
        generate=not args.no_generate,
    )
    table = to_markdown(rows, generate=not args.no_generate)
    print("\n" + table)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "# Ablation results\n\n"
            "Generated by `python -m eval.sweep`. `recall@k` and `nDCG@k` are computed against the\n"
            "golden set in `data/golden/qa.jsonl`, resolved onto whatever chunk ids each chunking\n"
            "configuration produced.\n\n" + table,
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
