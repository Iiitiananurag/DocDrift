#!/usr/bin/env python3
"""
DocDrift CLI (`cli.py`)

Command-line interface supporting:
    docdrift check --diff <file> --docs <dir> [--threshold 0.38] [--mock] [--json-out <file>] [--format terminal|json|both]

Accepts:
  - `--diff <file>`: Either a JSON file containing symbol-change event objects
    `[{symbol_name, change_type, old_value, new_value, file, line}, ...]` from the
    upstream diff parser, OR a raw unified git diff (`.diff` / `.patch`) which is
    automatically parsed into symbol-change events.
  - `--docs <dir_or_file>`: Either a JSON file containing parsed doc units
    `[{text, file, line_range, mentioned_symbols}, ...]` from the upstream doc parser,
    OR a directory of `.md` / `.rst` / `.txt` files (and/or `doc_units.json`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence,Tuple

# Support running both as `python matching/cli.py` and `python -m matching.cli`
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from matcher import match_changes_to_docs  # noqa: E402
from drift_classifier import classify_matched_pairs, DriftClassification  # noqa: E402


# ---------------------------------------------------------------------------
# ANSI Terminal Colors
# ---------------------------------------------------------------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"

    @classmethod
    def disable(cls) -> None:
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.CYAN = ""
        cls.MAGENTA = ""


# ---------------------------------------------------------------------------
# Input Loaders (JSON Upstream Structures + Raw Diff / Markdown Parsers)
# ---------------------------------------------------------------------------

def parse_unified_diff_fallback(diff_text: str, source_file: str = "changes.diff") -> List[Dict[str, Any]]:
    """
    Lightweight fallback parser when `--diff` is passed a raw unified `.diff` file
    instead of pre-parsed JSON from the upstream AST diff module.
    """
    events: List[Dict[str, Any]] = []
    current_file = source_file
    current_line = 1
    removed_lines: List[str] = []
    added_lines: List[Tuple[int, str]] = []

    def _flush_hunk() -> None:
        nonlocal removed_lines, added_lines
        if not removed_lines and not added_lines:
            return
        old_block = "\n".join(removed_lines).strip()
        new_block = "\n".join(text for _, text in added_lines).strip()
        line_no = added_lines[0][0] if added_lines else current_line

        # Identify symbol name from `def name`, `class name`, or `CONST_NAME =`
        combined = f"{new_block}\n{old_block}"
        sym_match = re.search(
            r"(?:def|class|async\s+def|function|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            r"|^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::[^=]+)?=",
            combined,
            flags=re.MULTILINE,
        )
        symbol_name = (sym_match.group(1) or sym_match.group(2)) if sym_match else "unknown_symbol"

        if old_block and not new_block:
            change_type = "removed"
        elif "def " in old_block and "def " in new_block:
            change_type = "signature_changed"
        else:
            change_type = "default_changed"

        events.append(
            {
                "symbol_name": symbol_name,
                "change_type": change_type,
                "old_value": old_block,
                "new_value": new_block,
                "file": current_file,
                "line": line_no,
            }
        )
        removed_lines = []
        added_lines = []

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            _flush_hunk()
            current_file = raw_line[6:].strip()
        elif raw_line.startswith("@@"):
            _flush_hunk()
            m = re.search(r"\+(\d+)", raw_line)
            if m:
                current_line = int(m.group(1))
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            removed_lines.append(raw_line[1:].strip())
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            added_lines.append((current_line, raw_line[1:].strip()))
            current_line += 1
        else:
            if removed_lines or added_lines:
                _flush_hunk()
            current_line += 1

    _flush_hunk()
    return events


def load_symbol_events(diff_path: Path) -> List[Dict[str, Any]]:
    """Loads code symbol-change events from a JSON file or unified diff."""
    content = diff_path.read_text(encoding="utf-8")
    if diff_path.suffix.lower() == ".json" or content.lstrip().startswith(("[", "{")):
        data = json.loads(content)
        if isinstance(data, dict):
            return list(data.get("symbol_events", data.get("events", [data])))
        return list(data)
    return parse_unified_diff_fallback(content, source_file=str(diff_path))


def parse_markdown_to_doc_units(md_path: Path, root_dir: Path) -> List[Dict[str, Any]]:
    """
    Splits a Markdown/text file into checkable paragraph/bullet units with
    `{text, file, line_range, mentioned_symbols}`.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    try:
        rel_path = str(md_path.relative_to(root_dir))
    except ValueError:
        rel_path = md_path.name

    units: List[Dict[str, Any]] = []
    buf: List[str] = []
    start_line = 1
    in_code_block = False

    def _flush(end_line: int) -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(s.strip() for s in buf if s.strip())
        buf = []
        if len(text) < 12 or text.startswith("#"):
            return
        # Extract backticked symbols and snake_case / camelCase tokens
        mentioned: List[str] = []
        for m in re.findall(r"`([^`]+)`", text):
            cleaned = m.strip().split("(")[0].strip()
            if cleaned and cleaned not in mentioned:
                mentioned.append(cleaned)
        for m in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b", text):
            if m not in mentioned:
                mentioned.append(m)
        units.append(
            {
                "text": text,
                "file": rel_path,
                "line_range": [start_line, end_line],
                "mentioned_symbols": mentioned,
            }
        )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not stripped or stripped.startswith("#"):
            _flush(idx - 1)
            start_line = idx + 1
            continue
        if stripped.startswith(("- ", "* ")) and buf:
            _flush(idx - 1)
            start_line = idx
        if not buf:
            start_line = idx
        buf.append(stripped.lstrip("-* ").strip())

    _flush(len(lines))
    return units


def load_doc_units(docs_path: Path) -> List[Dict[str, Any]]:
    """Loads checkable doc units from a JSON file or directory of docs."""
    if docs_path.is_file():
        content = docs_path.read_text(encoding="utf-8")
        if docs_path.suffix.lower() == ".json" or content.lstrip().startswith(("[", "{")):
            data = json.loads(content)
            if isinstance(data, dict):
                return list(data.get("doc_units", data.get("units", [data])))
            return list(data)
        return parse_markdown_to_doc_units(docs_path, docs_path.parent)

    # Directory mode: look for `doc_units.json` first, otherwise parse `.md`/`.rst`/`.txt`
    json_Direct = docs_path / "doc_units.json"
    if json_Direct.is_file():
        data = json.loads(json_Direct.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.get("doc_units", data.get("units", [])))
        return list(data)

    all_units: List[Dict[str, Any]] = []
    for ext in ("*.json", "*.md", "*.rst", "*.txt"):
        for fpath in sorted(docs_path.rglob(ext)):
            if fpath.suffix.lower() == ".json":
                try:
                    raw = json.loads(fpath.read_text(encoding="utf-8"))
                    items = raw.get("doc_units", raw) if isinstance(raw, dict) else raw
                    if isinstance(items, list) and items and isinstance(items[0], dict) and "text" in items[0]:
                        all_units.extend(items)
                except Exception:
                    continue
            else:
                all_units.extend(parse_markdown_to_doc_units(fpath, docs_path))
    return all_units


# ---------------------------------------------------------------------------
# Output Formatting (Colored Terminal + Structured JSON for CI)
# ---------------------------------------------------------------------------

def format_terminal_report(
    classifications: Sequence[DriftClassification],
    include_all: bool = False,
) -> str:
    lines: List[str] = []
    flagged = [c for c in classifications if c.verdict in ("likely outdated", "uncertain")]
    outdated_count = sum(1 for c in classifications if c.verdict == "likely outdated")
    uncertain_count = sum(1 for c in classifications if c.verdict == "uncertain")
    correct_count = sum(1 for c in classifications if c.verdict == "still correct")

    lines.append(
        f"{Colors.BOLD}{Colors.CYAN}=== DocDrift CI Check Report ==={Colors.RESET} "
        f"(pairs evaluated: {len(classifications)} | "
        f"{Colors.RED}likely outdated: {outdated_count}{Colors.RESET} | "
        f"{Colors.YELLOW}uncertain: {uncertain_count}{Colors.RESET} | "
        f"{Colors.GREEN}still correct: {correct_count}{Colors.RESET})"
    )

    items_to_show = classifications if include_all else flagged
    if not items_to_show:
        lines.append(f"{Colors.GREEN}✔ No documentation drift detected.{Colors.RESET}")
        return "\n".join(lines)

    for c in items_to_show:
        doc = c.doc_unit
        ev = c.symbol_event
        lr = doc.get("line_range", [1, 1])
        line_str = f"{lr[0]}-{lr[1]}" if lr[0] != lr[1] else f"{lr[0]}"
        doc_loc = f"{doc.get('file', 'README.md')}:{line_str}"
        code_loc = f"{ev.get('file', 'code')}:{ev.get('line', 1)}"
        match_method = doc.get("_match_method", "unknown")

        if c.verdict == "likely outdated":
            badge = f"{Colors.BOLD}{Colors.RED}[LIKELY OUTDATED]{Colors.RESET}"
        elif c.verdict == "uncertain":
            badge = f"{Colors.BOLD}{Colors.YELLOW}[UNCERTAIN]{Colors.RESET}"
        else:
            badge = f"{Colors.BOLD}{Colors.GREEN}[STILL CORRECT]{Colors.RESET}"

        lines.append(
            f"\n{badge} {Colors.BOLD}{doc_loc}{Colors.RESET} "
            f"(confidence: {Colors.MAGENTA}{c.confidence:.2f}{Colors.RESET}, "
            f"match_score: {c.match_score:.2f} via {Colors.CYAN}{match_method}{Colors.RESET})"
        )
        lines.append(f"  {Colors.DIM}Doc:{Colors.RESET}    \"{doc.get('text', '')}\"")
        lines.append(
            f"  {Colors.DIM}Change:{Colors.RESET} {ev.get('symbol_name')} ({ev.get('change_type')}) "
            f"at {code_loc}: `{ev.get('old_value')}` -> `{ev.get('new_value')}`"
        )
        lines.append(f"  {Colors.BOLD}Reason:{Colors.RESET} {c.reason}")

    return "\n".join(lines)


def build_json_payload(classifications: Sequence[DriftClassification]) -> Dict[str, Any]:
    flagged_units: List[Dict[str, Any]] = []
    all_units: List[Dict[str, Any]] = []

    for c in classifications:
        doc = c.doc_unit
        ev = c.symbol_event
        lr = doc.get("line_range", [1, 1])
        line_str = f"{lr[0]}-{lr[1]}" if lr[0] != lr[1] else f"{lr[0]}"
        record = {
            "doc_file": doc.get("file", "README.md"),
            "doc_line": line_str,
            "file_line": f"{doc.get('file', 'README.md')}:{line_str}",
            "doc_text": doc.get("text", ""),
            "symbol_name": ev.get("symbol_name", ""),
            "change_type": ev.get("change_type", ""),
            "old_value": ev.get("old_value", ""),
            "new_value": ev.get("new_value", ""),
            "code_file_line": f"{ev.get('file', '')}:{ev.get('line', 1)}",
            "match_method": doc.get("_match_method", ""),
            "match_score": c.match_score,
            "verdict": c.verdict,
            "reason": c.reason,
            "confidence": c.confidence,
            "provider": c.provider,
        }
        all_units.append(record)
        if c.verdict in ("likely outdated", "uncertain"):
            flagged_units.append(record)

    return {
        "summary": {
            "total_matched_pairs": len(classifications),
            "flagged_count": len(flagged_units),
            "likely_outdated_count": sum(1 for c in classifications if c.verdict == "likely outdated"),
            "uncertain_count": sum(1 for c in classifications if c.verdict == "uncertain"),
            "still_correct_count": sum(1 for c in classifications if c.verdict == "still correct"),
        },
        "flagged_units": flagged_units,
        "all_evaluated_pairs": all_units,
    }


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docdrift",
        description="DocDrift — CI check that flags documentation broken by code changes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_p = subparsers.add_parser(
        "check",
        help="Match code symbol-change events to doc units and classify documentation drift.",
    )
    check_p.add_argument(
        "--diff",
        required=True,
        type=Path,
        help="Path to symbol-change events JSON file or unified git diff (.diff/.patch).",
    )
    check_p.add_argument(
        "--docs",
        required=True,
        type=Path,
        help="Path to documentation directory (or parsed doc_units JSON file).",
    )
    check_p.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        help="Minimum similarity score threshold [0.0-1.0] (default: 0.38).",
    )
    check_p.add_argument(
        "--provider",
        choices=["groq", "openrouter"],
        default="groq",
        help="Free LLM API provider to use when not in mock mode (default: groq).",
    )
    check_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model ID (default: llama-3.1-8b-instant on Groq).",
    )
    check_p.add_argument(
        "--mock",
        action="store_true",
        help="Skip external LLM API call and use deterministic local drift classifier.",
    )
    check_p.add_argument(
        "--format",
        choices=["terminal", "json", "both"],
        default="both",
        help="Output format printed to stdout (default: both).",
    )
    check_p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional file path to write the JSON report (for CI PR comment consumption).",
    )
    check_p.add_argument(
        "--all",
        action="store_true",
        help="Include 'still correct' pairs in terminal output in addition to flagged units.",
    )
    check_p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in terminal output.",
    )
    check_p.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 if any 'likely outdated' doc units are flagged.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_color or os.environ.get("NO_COLOR"):
        Colors.disable()

    if not args.diff.exists():
        print(f"Error: --diff path not found: {args.diff}", file=sys.stderr)
        return 2
    if not args.docs.exists():
        print(f"Error: --docs path not found: {args.docs}", file=sys.stderr)
        return 2

    symbol_events = load_symbol_events(args.diff)
    doc_units = load_doc_units(args.docs)

    matched_pairs = match_changes_to_docs(
        symbol_events=symbol_events,
        doc_units=doc_units,
        similarity_threshold=args.threshold,
    )

    classifications = classify_matched_pairs(
        matched_pairs=matched_pairs,
        provider=args.provider,
        model=args.model,
        mock=args.mock,
    )

    json_payload = build_json_payload(classifications)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    if args.format in ("terminal", "both"):
        print(format_terminal_report(classifications, include_all=args.all))

    if args.format == "both":
        print(f"\n{Colors.BOLD}{Colors.CYAN}=== JSON Output (CI Consumption) ==={Colors.RESET}")
        print(json.dumps(json_payload, indent=2))
    elif args.format == "json":
        print(json.dumps(json_payload, indent=2))

    if args.fail_on_drift and json_payload["summary"]["likely_outdated_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())