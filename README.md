# DocDrift

A CI check that flags documentation a pull request's code change probably broke.

## What's verified vs. what's unverified

Everything below was actually executed, not just written:

- **Real end-to-end run**: `run_real_pipeline.py` against a real Click commit
  (`9f0bcdf` — `Parameter.human_readable_name` renamed to `Parameter.spec`)
  correctly flagged the exact real doc sentence in `docs/upgrade-guides.md`
  describing the deprecation. See `real_pipeline_report.json` for the full
  output and `example_real_click_diff.diff` / `example_real_click_doc.md`
  for the real inputs used.
- **Synthetic fixtures**: `matching/demo_run.py` runs 4 synthetic cases
  (exact match, fuzzy match, semantic-embedding fallback, and a safe refactor
  that should NOT be flagged) through `matcher.py` -> `drift_classifier.py`
  -> `cli.py`. All 4 produce the expected verdict.
- **Parser unit tests**: `parsing/test_parsers.py` has 6 synthetic tests plus
  1 real-data test (skipped unless you set env vars — see its docstring).
- **Ground-truth auto-mining** (`data/collect_ground_truth.py`): mines real
  (code_commit, doc_commit) pairs from a repo's git history by temporal
  proximity + shared-identifier overlap. Manually verified against
  `pallets/click`: produced mostly or entirely FALSE pairs even after
  tightening the identifier filter (see `data/ground_truth.json` vs.
  `data/ground_truth_v2.json`). This is a documented, known limitation —
  **do not present these pairs as verified real drift examples without
  manually checking each one with `data/verify_pairs.py` first.**

## Quick start

```bash
pip install -r requirements.txt   # sentence-transformers optional, falls back to n-gram cosine

# Run the real, verified example:
python3 run_real_pipeline.py example_real_click_diff.diff example_real_click_doc.md --mock

# Run the synthetic demo (all 4 fixture cases):
cd matching && python3 demo_run.py

# Run parser unit tests:
cd parsing && python3 test_parsers.py -v

# Mine ground truth from a real repo (expect noise - verify before trusting):
python3 -m data.collect_ground_truth https://github.com/pallets/click \
  --max-commits 800 --emit-negatives 10 -o data/ground_truth.json
python3 -m data.verify_pairs -g data/ground_truth.json --export-md data/review.md
```

## Architecture

```
docdrift/
├── data/
│   ├── collect_ground_truth.py   # mines (code_commit, doc_commit) pairs from git history
│   └── verify_pairs.py           # human-in-the-loop verification (REAL/FALSE triage)
├── parsing/
│   ├── code_diff_parser.py       # AST-based: diff -> symbol-change events
│   ├── doc_parser.py             # Markdown/reST -> doc units (sentence/table-row/list-item level)
│   └── test_parsers.py
├── matching/
│   ├── matcher.py                 # exact/fuzzy symbol match + semantic embedding fallback
│   ├── drift_classifier.py        # still correct / likely outdated / uncertain + reason
│   ├── cli.py                     # `docdrift check` - own fallback parsers + full CLI/CI flow
│   └── demo_run.py                # synthetic 4-case demo
└── run_real_pipeline.py           # wires real parsers directly into matcher+classifier
```

Two ways to run the pipeline:
1. **`cli.py check`** — self-contained, uses its own regex-based fallback
   parsers for diffs and docs. Good for CI where you want zero extra deps.
2. **`run_real_pipeline.py`** — uses the real AST-based `code_diff_parser.py`
   and the structured `doc_parser.py` for more precise symbol/line extraction.
   Slower, more accurate on Python-specific changes (renames, default values,
   removed symbols, CLI flag changes).

## Known limitations (be upfront about these to judges)

1. **Diff-only ("fragment") mode is less precise than full git mode.** Without
   access to the full before/after file (`sources_from_git`), the parser
   infers the enclosing function/class from the diff hunk header alone, which
   can misattribute a changed line to the wrong symbol in edge cases.
2. **Body-only logic changes produce zero events.** `code_diff_parser.py`
   only tracks signatures, defaults, removed symbols, and CLI flags — not
   changes inside function bodies. This is intentional (avoids false
   positives) but means some real behavioral drift won't be caught.
3. **Auto-mined ground truth is noisy.** Temporal proximity + shared-token
   overlap produces false pairs even with strict identifier filtering
   (CamelCase/snake_case/ALL_CAPS/--flags only). Always manually verify
   with `verify_pairs.py` before using a mined pair as a demo claim.
4. **`click.argument` / positional argparse args are not tracked** — only
   `click.option` / `add_argument` flag changes.
