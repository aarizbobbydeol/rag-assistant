"""Fail CI when the refusal metrics regress.

The eval prints numbers; nothing was stopping a change from quietly making them
worse. These two are the ones a hallucination guard lives or dies by, so they
get a hard floor rather than a graph someone might look at.

Thresholds are deliberately set at the current measured values, not at
aspirational ones: the point is to catch a regression on the next commit, not to
describe an ambition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATES = {
    # Answering a question the corpus cannot answer is the failure this whole
    # project exists to prevent, so it gets no slack at all.
    "false_answer_rate": ("<=", 0.0),
    # Refusing a question the corpus does answer is the cost side of the trade.
    # Pinned just above the measured 0.017 so a single new refusal fails.
    "over_refusal_rate": ("<=", 0.02),
    # Retrieval feeds everything else; a drop here invalidates the rest.
    "recall@k": (">=", 0.98),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON written by eval.run_eval --json-out")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))

    found: dict[str, float] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in GATES and isinstance(value, (int, float)):
                    found.setdefault(key, float(value))
                walk(value)

    walk(report)

    missing = sorted(set(GATES) - set(found))
    if missing:
        print(f"FAIL: metrics absent from the report: {', '.join(missing)}")
        return 1

    failures = []
    for metric, (op, limit) in GATES.items():
        value = found[metric]
        ok = value <= limit if op == "<=" else value >= limit
        print(f"  {'ok  ' if ok else 'FAIL'} {metric:20} {value:.3f}  (gate {op} {limit})")
        if not ok:
            failures.append(f"{metric} = {value:.3f}, gate is {op} {limit}")

    if failures:
        print("\nEval gate failed:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("\nEval gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
