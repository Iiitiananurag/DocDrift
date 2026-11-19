#!/usr/bin/env python3
"""
run_real_pipeline.py

End-to-end wiring: real code_diff_parser + doc_parser output feeds directly
into matcher.match_changes_to_docs() and drift_classifier.classify_matched_pairs(),
bypassing cli.py's own fallback parsers.

Usage:
    python3 run_real_pipeline.py <diff_file> <doc_file> [--mock]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "parsing"))
sys.path.insert(0, str(Path(__file__).parent / "matching"))

import code_diff_parser as cdp  # noqa: E402
import doc_parser as dp         # noqa: E402
from matcher import match_changes_to_docs  # noqa: E402
from drift_classifier import classify_matched_pairs  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_real_pipeline.py <diff_file> <doc_file> [--mock]")
        return 1

    diff_path = Path(sys.argv[1])
    doc_path = Path(sys.argv[2])
    use_mock = "--mock" in sys.argv

    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace")

    # 1. Parse the real diff into symbol-change events (fragment mode - no git repo needed here)
    events = cdp.extract_events(diff_text)
    print(f"=== {len(events)} symbol-change events from {diff_path.name} ===")
    for e in events:
        print(f"  {e['change_type']:20s} {e['symbol_name']}")

    # 2. Parse the real doc file into doc units
    units = dp.parse_text(doc_text, str(doc_path))
    print(f"\n=== {len(units)} doc units from {doc_path.name} ===")

    if not events:
        print("\nNo symbol events extracted - nothing to match. (Expected if the diff is body-only logic.)")
        return 0

    # 3. Match events to doc units (real matcher: exact/fuzzy symbol + semantic embedding fallback)
    matched_pairs = match_changes_to_docs(events, units)
    print(f"\n=== {len(matched_pairs)} matched (doc_unit, symbol_event, score) pairs above threshold ===")
    for doc, ev, score in matched_pairs:
        print(f"  score={score:.3f}  method={doc.get('_match_method')}  "
              f"symbol={ev['symbol_name']}  doc={doc['file']}:{doc['line_range']}")

    if not matched_pairs:
        print("\nNo matches above threshold - nothing to classify.")
        return 0

    # 4. Classify each matched pair as still correct / likely outdated / uncertain
    results = classify_matched_pairs(matched_pairs, mock=use_mock)
    print(f"\n=== {len(results)} classifications (mock={use_mock}) ===")
    flagged = []
    for r in results:
        marker = "🔴" if r.verdict == "likely outdated" else ("🟡" if r.verdict == "uncertain" else "🟢")
        print(f"  {marker} [{r.verdict:16s} conf={r.confidence:.2f}] "
              f"{r.doc_unit['file']}:{r.doc_unit['line_range']}")
        print(f"      doc text: {r.doc_unit['text'][:100]!r}")
        print(f"      reason:   {r.reason}")
        if r.verdict in ("likely outdated", "uncertain"):
            flagged.append(r.to_dict())

    out_path = Path("real_pipeline_report.json")
    out_path.write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding="utf-8")
    print(f"\nWrote full report to {out_path} ({len(flagged)} flagged of {len(results)} classified)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
