#!/usr/bin/env python3
"""
Synthetic Demo Runner for DocDrift (`demo_run.py`)

Creates 3 realistic synthetic code-diff + documentation pairs covering:
  Pair 1 (Exact Symbol Match -> Likely Outdated):
    `connect_db` default `timeout` changed from `5` to `30` seconds in `src/db.py:42`,
    while `docs/database.md:14-16` still states the default timeout is 5 seconds.
  Pair 2 (Fuzzy Symbol Match -> Likely Outdated):
    `MAX_RETRY_ATTEMPTS` changed from `3` to `5` in `src/http_client.py:18`, while
    `docs/configuration.md:28-30` mentions `max_retry_attempt` (singular) and says
    failed requests are retried 3 times.
  Pair 3 (Semantic Embedding Fallback + Control Pair):
    - 3a (`Semantic Embedding Match -> Likely Outdated`): `LegacyTokenAuth` was
      removed (`src/auth.py:95`), and `docs/security.md:52-55` describes authenticating
      with bearer tokens without naming the exact symbol in `mentioned_symbols`.
    - 3b (`Exact Symbol Match -> Still Correct` Control): ` flush_batch` internal
      buffer type hint changed in `src/batch.py:61`, while `docs/ingestion.md:10-12`
      remains accurate.

Writes the synthetic fixtures to `matching/demo_fixtures/` and invokes
`cli.py check --diff ... --docs ... --mock --all` to demonstrate both the colored
terminal output and the CI JSON payload.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = THIS_DIR / "demo_fixtures"


SYNTHETIC_SYMBOL_EVENTS = [
    {
        "symbol_name": "connect_db",
        "change_type": "default_changed",
        "old_value": "def connect_db(dsn: str, timeout: int = 5) -> Connection",
        "new_value": "def connect_db(dsn: str, timeout: int = 30) -> Connection",
        "file": "src/db.py",
        "line": 42,
    },
    {
        "symbol_name": "MAX_RETRY_ATTEMPTS",
        "change_type": "default_changed",
        "old_value": "MAX_RETRY_ATTEMPTS = 3",
        "new_value": "MAX_RETRY_ATTEMPTS = 5",
        "file": "src/http_client.py",
        "line": 18,
    },
    {
        "symbol_name": "LegacyTokenAuth",
        "change_type": "removed",
        "old_value": "class LegacyTokenAuth(AuthProvider): header = 'X-Legacy-Token'",
        "new_value": "",
        "file": "src/auth.py",
        "line": 95,
    },
    {
        "symbol_name": "flush_batch",
        "change_type": "internal_refactor",
        "old_value": "def flush_batch(items: list) -> int",
        "new_value": "def flush_batch(items: Sequence[Record]) -> int",
        "file": "src/batch.py",
        "line": 61,
    },
]

SYNTHETIC_DOC_UNITS = [
    {
        "text": "Call `connect_db(dsn)` to open a pooled session; if the server does not respond within the default 5 seconds timeout, a ConnectionTimeoutError is raised.",
        "file": "docs/database.md",
        "line_range": [14, 16],
        "mentioned_symbols": ["connect_db", "ConnectionTimeoutError"],
    },
    {
        "text": "Transient 503 responses are automatically retried up to 3 times as controlled by `max_retry_attempt` before surfacing an error to the caller.",
        "file": "docs/configuration.md",
        "line_range": [28, 30],
        "mentioned_symbols": ["max_retry_attempt"],
    },
    {
        "text": "For older v1 integrations, authenticate requests by passing your static key in the `X-Legacy-Token` header via the legacy token authentication provider.",
        "file": "docs/security.md",
        "line_range": [52, 55],
        "mentioned_symbols": [],  # Intentionally empty to exercise semantic embedding fallback!
    },
    {
        "text": "Use `flush_batch(items)` to flush pending records to storage and return the count of written items.",
        "file": "docs/ingestion.md",
        "line_range": [10, 12],
        "mentioned_symbols": ["flush_batch"],
    },
]


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    diff_json_path = FIXTURES_DIR / "symbol_events.json"
    docs_dir = FIXTURES_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_units_path = docs_dir / "doc_units.json"
    report_json_path = FIXTURES_DIR / "demo_report.json"

    diff_json_path.write_text(json.dumps(SYNTHETIC_SYMBOL_EVENTS, indent=2), encoding="utf-8")
    doc_units_path.write_text(json.dumps(SYNTHETIC_DOC_UNITS, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(THIS_DIR / "cli.py"),
        "check",
        "--diff",
        str(diff_json_path),
        "--docs",
        str(docs_dir),
        "--mock",
        "--all",
        "--json-out",
        str(report_json_path),
    ]
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())