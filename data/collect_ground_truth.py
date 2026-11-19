#!/usr/bin/env python3
"""collect_ground_truth.py (v2 - strict identifier matching)"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import git  # type: ignore
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False


DOC_EXTENSIONS: Set[str] = {".md", ".rst", ".txt", ".adoc", ".mdx"}
DOC_DIRS: Set[str] = {"docs", "doc", "documentation"}
DOC_BASENAMES: Set[str] = {
    "readme", "changelog", "changes", "history", "contributing", "authors",
    "mkdocs.yml", "readthedocs.yml", ".readthedocs.yaml", ".readthedocs.yml",
}
NON_RUNTIME_PREFIXES: Tuple[str, ...] = (
    ".github/", ".circleci/", ".pre-commit", ".gitignore", ".flake8", ".editorconfig",
)
CODE_EXTENSIONS: Set[str] = {
    ".py", ".pyi", ".c", ".h", ".cpp", ".rs", ".go", ".js", ".ts",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".sh",
}
DEFAULT_DOC_MSG_REGEX = (
    r"(?i)(?:"
    r"^\s*docs?(?:\([^)]+\))?\s*[:\-]\s*(?:update|fix|sync|add|remove|document|clarify|reflect|correct|adjust|mention|rename|deprecate)"
    r"|"
    r"\b(?:update|fix|sync|clarify|correct|add|remove|document)\b[^\n]{0,40}\b(?:docs?|documentation|readme|docstring|api\.md|changelog)\b"
    r"|"
    r"\b(?:docs?|documentation|readme)\b[^\n]{0,40}\b(?:update|fix|sync|for|after|missing|outdated|reflect|clarify)\b"
    r")"
)
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
URL_RE = re.compile(r"https?://[^\s)\]'\"<>]+")
CLI_FLAG_RE = re.compile(r"(?<![a-zA-Z0-9_-])--[a-zA-Z][a-zA-Z0-9_-]{1,}(?![a-zA-Z0-9_-])")
DOTTED_PATH_RE = re.compile(
    r"(?<![a-zA-Z0-9_.-])[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]+)+(?![a-zA-Z0-9_.-])"
)
WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
SNAKE_CASE_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*[a-zA-Z0-9]$")
CAMEL_CASE_RE = re.compile(
    r"^(?:[a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*|[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*|[A-Z]{2,}[a-z][a-zA-Z0-9]*)$"
)
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

GENERIC_CLI_FLAGS: Set[str] = {"--help", "--version"}
IGNORED_DOTTED_SUFFIXES: Set[str] = {
    "py", "pyi", "pyc", "md", "rst", "txt", "adoc", "mdx", "yml", "yaml",
    "toml", "json", "ini", "cfg", "sh", "bat", "ps1", "html", "htm", "css",
    "js", "ts", "jsx", "tsx", "png", "jpg", "jpeg", "gif", "svg", "ico",
    "pdf", "zip", "gz", "tar", "whl", "lock", "in", "c", "h", "cpp", "rs",
    "go", "java", "rb", "php", "xml", "csv", "sql", "log",
    "com", "org", "io", "net", "dev", "edu", "gov", "ai", "app", "co", "uk", "de",
}
IGNORED_DOTTED_PREFIXES: Set[str] = {
    "t", "te", "typing", "typing_extensions", "cabc", "collections", "abc",
    "self", "cls", "e", "i",
}
GENERIC_IDENTIFIERS: Set[str] = {
    "TYPE_CHECKING", "TypeVar", "ParamSpec", "TypeAlias", "TypeGuard", "TypeIs",
    "ClassVar", "NotRequired", "TypedDict", "NamedTuple", "Generic", "Protocol",
    "Final", "Literal", "Annotated", "Unpack", "Concatenate", "NoReturn",
    "ReadOnly", "Required", "Never",
    "GitHub", "GitLab", "PyPI", "ReadTheDocs", "MacOS", "macOS", "iOS",
    "JavaScript", "TypeScript", "PowerShell", "PyTest", "Sphinx",
    "foo_bar", "my_func", "my_app", "my_cmd", "my_cli", "my_option",
    "NOTE", "TODO", "FIXME", "XXX", "HACK", "WARNING", "IMPORTANT", "CAUTION",
    "TIP", "DEPRECATED", "HTTP", "HTTPS", "HTML", "JSON", "YAML", "TOML",
    "XML", "CSV", "SQL", "API", "URL", "URI", "CLI", "GUI", "TUI", "UTF",
    "ASCII", "EOF", "EOL", "CRLF", "POSIX", "UNIX", "LINUX", "WINDOWS",
    "OSX", "BSD", "MIT", "GPL", "GNU", "RST", "MDX", "PEP", "RFC", "SSL",
    "TLS", "TCP", "UDP", "DNS", "SSH", "FTP", "GET", "POST", "PUT", "DELETE",
    "PATCH", "HEAD", "OPTIONS", "TRUE", "FALSE", "NONE", "NULL", "NAN", "INF",
    "ANY", "ALL", "END", "NEW", "OLD", "MAX", "MIN", "ENV", "VAR", "STR",
    "INT", "BOOL", "SET", "MAP", "KEY", "VAL", "ERR", "LOG", "SRC", "DST",
    "DIR", "TMP", "PWD", "UID", "GID", "PID", "TTY", "CPU", "GPU", "RAM",
    "SDK", "IDE", "VCS", "GIT", "PIP", "PYPI", "WIP", "TBD", "TBA", "FYI",
    "FAQ", "TLDR", "LICENSE", "README", "CHANGELOG", "HISTORY", "AUTHORS",
    "NOTICE", "VERSION", "DEFAULT", "EXAMPLE", "OUTPUT", "INPUT", "ERROR",
}


def extract_code_identifiers(text: str) -> Set[str]:
    cleaned = URL_RE.sub(" ", text)
    found: Set[str] = set()
    for flag in CLI_FLAG_RE.findall(cleaned):
        if flag.lower() not in GENERIC_CLI_FLAGS:
            found.add(flag)
    for path in DOTTED_PATH_RE.findall(cleaned):
        parts = path.split(".")
        if parts[0] in IGNORED_DOTTED_PREFIXES:
            continue
        if parts[-1].lower() in IGNORED_DOTTED_SUFFIXES:
            continue
        if path not in GENERIC_IDENTIFIERS:
            found.add(path)
    for raw_tok in WORD_TOKEN_RE.findall(cleaned):
        tok = raw_tok.strip("_")
        if len(tok) < 3 or tok in GENERIC_IDENTIFIERS:
            continue
        if SNAKE_CASE_RE.match(tok) and len(tok) >= 4:
            found.add(tok)
        elif CAMEL_CASE_RE.match(tok):
            found.add(tok)
        elif ALL_CAPS_RE.match(tok) and sum(c.isalpha() for c in tok) >= 2:
            found.add(tok)
    return found


@dataclasses.dataclass
class CommitMeta:
    sha: str
    timestamp: int
    message: str
    files_changed: List[str]


def is_doc_file(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("./")
    parts = norm.lower().split("/")
    p = Path(norm.lower())
    if any(part in DOC_DIRS for part in parts[:-1]):
        return True
    if p.suffix in DOC_EXTENSIONS:
        return True
    if p.stem in DOC_BASENAMES or p.name in DOC_BASENAMES:
        return True
    return False


def is_changelog_file(path: str) -> bool:
    stem = Path(path.lower()).stem
    return stem in {"changelog", "changes", "history", "news", "releases"}


def is_test_or_ci_file(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("./").lower()
    if norm.startswith(NON_RUNTIME_PREFIXES):
        return True
    parts = norm.split("/")
    if any(p in {"test", "tests", "testing"} for p in parts[:-1]):
        return True
    basename = Path(norm).name
    return basename.startswith("test_") or basename.endswith("_test.py")


def is_code_file(path: str) -> bool:
    if is_doc_file(path):
        return False
    p = Path(path.lower())
    return p.suffix in CODE_EXTENSIONS


def classify_commit(files, include_test_commits=False, ignore_changelog_only=True) -> str:
    if not files:
        return "mixed_or_other"
    doc_files = [f for f in files if is_doc_file(f)]
    if len(doc_files) == len(files):
        if ignore_changelog_only and all(is_changelog_file(f) for f in doc_files):
            return "mixed_or_other"
        return "doc_only"
    if len(doc_files) == 0:
        code_files = [f for f in files if is_code_file(f)]
        if not code_files:
            return "mixed_or_other"
        if not include_test_commits:
            runtime_code = [f for f in code_files if not is_test_or_ci_file(f)]
            if not runtime_code:
                return "mixed_or_other"
        return "code_only"
    return "mixed_or_other"


def run_git(repo_dir: Path, args: Sequence[str]) -> str:
    cmd = ["git", "-C", str(repo_dir), *args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git command failed ({' '.join(cmd)}): {proc.stderr.strip()}")
    return proc.stdout


def ensure_local_clone(repo_url_or_path: str, cache_dir=None, max_commits=None) -> Tuple[Path, str, bool]:
    candidate = Path(repo_url_or_path).expanduser()
    if candidate.exists() and (candidate / ".git").exists():
        try:
            remote = run_git(candidate, ["config", "--get", "remote.origin.url"]).strip()
        except RuntimeError:
            remote = str(candidate.resolve())
        return candidate.resolve(), remote or str(candidate.resolve()), False
    repo_slug = repo_url_or_path.rstrip("/").split("/")[-2:]
    slug_name = "__".join(s.replace(".git", "") for s in repo_slug)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_dir = cache_dir / slug_name
        if (target_dir / ".git").exists():
            return target_dir, repo_url_or_path, False
        is_temp = False
    else:
        target_dir = Path(tempfile.mkdtemp(prefix=f"docdrift_{slug_name}_"))
        is_temp = True
    if HAS_GITPYTHON:
        opts = ["--no-checkout", "--filter=blob:none"]
        if max_commits and max_commits > 0:
            opts.append(f"--depth={max_commits}")
        git.Repo.clone_from(repo_url_or_path, str(target_dir), multi_options=opts)
    else:
        clone_args = ["clone", "--no-checkout", "--filter=blob:none"]
        if max_commits and max_commits > 0:
            clone_args.extend(["--depth", str(max_commits)])
        clone_args.extend([repo_url_or_path, str(target_dir)])
        subprocess.run(["git", *clone_args], check=True)
    return target_dir, repo_url_or_path, is_temp


def load_commits(repo_dir: Path, repo_url: str, max_commits: int = 2000, first_parent_only: bool = False) -> List[CommitMeta]:
    sep = "===DOCDRIFT_COMMIT==="
    field_sep = "===DOCDRIFT_FIELD==="
    log_args = ["log", "--no-merges", f"-n{max_commits}", f"--pretty=format:{sep}%H{field_sep}%ct{field_sep}%B{field_sep}", "--name-only"]
    if first_parent_only:
        log_args.insert(1, "--first-parent")
    raw_output = run_git(repo_dir, log_args)
    raw_blocks = [b.strip() for b in raw_output.split(sep) if b.strip()]
    commits: List[CommitMeta] = []
    for block in raw_blocks:
        parts = block.split(field_sep)
        if len(parts) < 4:
            continue
        sha = parts[0].strip()
        ts_str = parts[1].strip()
        message = parts[2].strip()
        files_block = parts[3].strip()
        files = [line.strip() for line in files_block.splitlines() if line.strip()]
        if not ts_str or not message:
            continue
        commits.append(CommitMeta(sha=sha, timestamp=int(ts_str), message=message, files_changed=files))
    commits.reverse()
    return commits


def extract_diff_details(repo_dir: Path, commit_sha: str, files: Sequence[str]):
    try:
        diff_text = run_git(repo_dir, ["show", "--format=", "-U0", commit_sha, "--", *files])
    except RuntimeError:
        return set(), {}, {}
    tokens: Set[str] = set()
    pre_ranges: Dict[str, List[List[int]]] = {}
    post_ranges: Dict[str, List[List[int]]] = {}
    current_file: Optional[str] = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if line.startswith("--- a/"):
            if current_file is None:
                current_file = line[6:].strip()
            continue
        m = HUNK_HEADER_RE.match(line)
        if m and current_file:
            old_start = int(m.group(1)); old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3)); new_count = int(m.group(4)) if m.group(4) is not None else 1
            old_end = old_start if old_count == 0 else (old_start + old_count - 1)
            new_end = new_start if new_count == 0 else (new_start + new_count - 1)
            pre_ranges.setdefault(current_file, []).append([old_start, old_end])
            post_ranges.setdefault(current_file, []).append([new_start, new_end])
            continue
        if (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---")):
            tokens.update(extract_code_identifiers(line[1:]))
    return tokens, pre_ranges, post_ranges


def format_date_gap(seconds: int) -> str:
    days = seconds / 86400.0
    if abs(days) >= 1.0:
        return f"{days:.2f} days"
    hours = seconds / 3600.0
    if abs(hours) >= 1.0:
        return f"{hours:.2f} hours"
    return f"{seconds/60.0:.1f} minutes"


def find_ground_truth_pairs(repo_url_or_path, max_commits=2500, max_commit_gap=15, max_days=14.0, min_shared_tokens=1, doc_msg_pattern=DEFAULT_DOC_MSG_REGEX, include_test_commits=False, cache_dir=None, emit_negatives=0):
    repo_dir, canonical_repo, is_temp = ensure_local_clone(repo_url_or_path, cache_dir=cache_dir, max_commits=max_commits)
    msg_regex = re.compile(doc_msg_pattern)
    max_seconds = int(max_days * 86400)
    try:
        commits = load_commits(repo_dir, canonical_repo, max_commits=max_commits)
        print(f"  -> loaded {len(commits)} non-merge commits", file=sys.stderr)
        classifications = [classify_commit(c.files_changed, include_test_commits=include_test_commits) for c in commits]
        from collections import Counter
        print(f"  -> classification breakdown: {dict(Counter(classifications))}", file=sys.stderr)
        diff_cache = {}
        def get_diff_info(c):
            if c.sha not in diff_cache:
                diff_cache[c.sha] = extract_diff_details(repo_dir, c.sha, c.files_changed)
            return diff_cache[c.sha]
        pairs = []
        paired_code_shas = set()
        commits_near_any_doc = set()
        for idx, commit_b in enumerate(commits):
            if classifications[idx] == "doc_only":
                for back in range(max(0, idx - max_commit_gap * 2), idx):
                    commits_near_any_doc.add(commits[back].sha)
            if classifications[idx] != "doc_only":
                continue
            first_line = commit_b.message.splitlines()[0] if commit_b.message else ""
            if not msg_regex.search(first_line) and not msg_regex.search(commit_b.message):
                continue
            doc_tokens, doc_pre_ranges, doc_post_ranges = get_diff_info(commit_b)
            doc_tokens = doc_tokens.union(extract_code_identifiers(commit_b.message))
            best_candidate = None
            start_idx = max(0, idx - max_commit_gap)
            for cand_idx in range(idx - 1, start_idx - 1, -1):
                commit_a = commits[cand_idx]
                if classifications[cand_idx] != "code_only":
                    continue
                time_delta = commit_b.timestamp - commit_a.timestamp
                if time_delta < 0 or time_delta > max_seconds:
                    continue
                code_tokens, _, _ = get_diff_info(commit_a)
                shared = sorted(doc_tokens.intersection(code_tokens))
                if len(shared) < min_shared_tokens:
                    continue
                commit_distance = idx - cand_idx
                score_key = (len(shared), -commit_distance)
                if best_candidate is None or score_key > (best_candidate[0], -best_candidate[1]):
                    best_candidate = (len(shared), commit_distance, commit_a, shared)
            if best_candidate is not None:
                _, commit_dist, commit_a, shared_tokens = best_candidate
                delta_sec = max(0, commit_b.timestamp - commit_a.timestamp)
                paired_code_shas.add(commit_a.sha)
                pairs.append({
                    "repo": canonical_repo,
                    "code_commit_sha": commit_a.sha,
                    "doc_commit_sha": commit_b.sha,
                    "code_files_changed": commit_a.files_changed,
                    "doc_files_changed": commit_b.files_changed,
                    "doc_commit_message": commit_b.message.splitlines()[0].strip(),
                    "date_gap": format_date_gap(delta_sec),
                    "date_gap_days": round(delta_sec / 86400.0, 4),
                    "commit_distance": commit_dist,
                    "shared_identifiers": shared_tokens[:15],
                    "doc_changed_line_ranges": doc_pre_ranges,
                    "doc_post_line_ranges": doc_post_ranges,
                })
        negatives = []
        if emit_negatives > 0:
            for idx, c in enumerate(commits):
                if classifications[idx] == "code_only" and c.sha not in paired_code_shas and c.sha not in commits_near_any_doc:
                    negatives.append({"repo": canonical_repo, "code_commit_sha": c.sha, "code_files_changed": c.files_changed, "code_commit_message": c.message.splitlines()[0].strip()})
                    if len(negatives) >= emit_negatives:
                        break
        return pairs, negatives
    finally:
        if is_temp and repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mine ground-truth pairs for DocDrift.")
    parser.add_argument("repos", nargs="+")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/ground_truth.json"))
    parser.add_argument("--max-commits", type=int, default=2500)
    parser.add_argument("--max-commit-gap", type=int, default=15)
    parser.add_argument("--max-days", type=float, default=14.0)
    parser.add_argument("--min-shared-tokens", type=int, default=1)
    parser.add_argument("--doc-regex", type=str, default=DEFAULT_DOC_MSG_REGEX)
    parser.add_argument("--cache-dir", type=Path, default=Path(".docdrift_repo_cache"))
    parser.add_argument("--emit-negatives", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    all_pairs, all_negatives = [], []
    for repo in args.repos:
        print(f"[DocDrift] Scanning {repo} ...", file=sys.stderr)
        pairs, negatives = find_ground_truth_pairs(
            repo_url_or_path=repo, max_commits=args.max_commits, max_commit_gap=args.max_commit_gap,
            max_days=args.max_days, min_shared_tokens=args.min_shared_tokens, doc_msg_pattern=args.doc_regex,
            cache_dir=args.cache_dir, emit_negatives=args.emit_negatives,
        )
        print(f"[DocDrift]   Found {len(pairs)} ground-truth pairs in {repo}", file=sys.stderr)
        all_pairs.extend(pairs)
        all_negatives.extend(negatives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_pairs, indent=2) + "\n", encoding="utf-8")
    print(f"[DocDrift] Wrote {len(all_pairs)} total pairs to {args.output}", file=sys.stderr)
    if args.emit_negatives > 0:
        neg_path = args.output.with_suffix(".negatives.json")
        neg_path.write_text(json.dumps(all_negatives, indent=2) + "\n", encoding="utf-8")
        print(f"[DocDrift] Wrote {len(all_negatives)} negative control commits to {neg_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
