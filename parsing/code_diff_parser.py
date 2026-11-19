#!/usr/bin/env python3
"""DocDrift code_diff_parser: unified git diff of Python files -> symbol-change events.

Event schema (exactly these six keys):
    {symbol_name, change_type, old_value, new_value, file, line}

change_type values:
    renamed, removed, added                      (functions / classes; symbol_name is the qualified name)
    param_renamed, param_removed, param_added    (symbol_name = "func.param")
    default_changed                              (symbol_name = "func.param"; old/new = unparsed source)
    flag_renamed, flag_alias_added, flag_alias_removed, flag_added, flag_removed,
    flag_default_changed, flag_type_changed, flag_required_changed,
    flag_envvar_changed, flag_is_flag_changed, flag_action_changed,
    flag_choices_changed, flag_nargs_changed     (click.option / parser.add_argument)

Two modes, chosen per file:
  * FULL mode  - old and new file contents are available (pass dicts, or use --repo/--head).
                 Both versions are parsed with `ast` and the symbol tables are compared. Exact.
  * FRAGMENT mode - only the diff is available. Each hunk is rebuilt into old/new text and parsed
                 with `ast` after trimming lines until it parses. If a hunk cuts through a
                 multi-line signature, a parameter-list fallback is used and the owner is guessed
                 from the hunk header. Less exact: prefer FULL mode.

Line numbers: new-file line for everything except removals (old-file line).
Requires Python 3.9+ (ast.unparse).
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
_MAX_TRIM = 30
_FLAG_ATTRS_CLICK = ("default", "type", "required", "envvar", "is_flag")
_FLAG_ATTRS_ARGPARSE = ("default", "type", "required", "action", "choices", "nargs")


# --------------------------------------------------------------------------- diff parsing
def _strip_prefix(p: str) -> str | None:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_unified_diff(diff_text: str) -> list[dict]:
    """-> [{old_path, new_path, hunks:[{ctx, old:[(lineno,text)], new:[(lineno,text)]}]}]"""
    files: list[dict] = []
    cur = None
    hunk = None
    old_rem = new_rem = old_no = new_no = 0
    for raw in diff_text.splitlines():
        if hunk is not None and (old_rem > 0 or new_rem > 0):
            tag, body = (raw[:1], raw[1:]) if raw else (" ", "")
            if tag == "\\":
                continue
            if tag == "-":
                hunk["old"].append((old_no, body)); hunk["old_changed"].add(old_no)
                old_no += 1; old_rem -= 1
            elif tag == "+":
                hunk["new"].append((new_no, body)); hunk["new_changed"].add(new_no)
                new_no += 1; new_rem -= 1
            else:
                hunk["old"].append((old_no, body)); hunk["new"].append((new_no, body))
                old_no += 1; new_no += 1; old_rem -= 1; new_rem -= 1
            continue
        hunk = None
        if raw.startswith("diff --git "):
            cur = {"old_path": None, "new_path": None, "hunks": [], "_plus": False}
            files.append(cur)
            m = re.match(r"diff --git a/(.*?) b/(.*)$", raw)
            if m:
                cur["old_path"], cur["new_path"] = m.group(1), m.group(2)
        elif raw.startswith("rename from ") and cur:
            cur["old_path"] = raw[len("rename from "):]
        elif raw.startswith("rename to ") and cur:
            cur["new_path"] = raw[len("rename to "):]
        elif raw.startswith("--- "):
            if cur is None or cur["_plus"]:
                cur = {"old_path": None, "new_path": None, "hunks": [], "_plus": False}
                files.append(cur)
            cur["old_path"] = _strip_prefix(raw[4:])
        elif raw.startswith("+++ ") and cur is not None:
            cur["new_path"] = _strip_prefix(raw[4:])
            cur["_plus"] = True
        else:
            m = _HUNK_RE.match(raw)
            if m and cur is not None:
                old_no, new_no = int(m.group(1)), int(m.group(3))
                old_rem = int(m.group(2)) if m.group(2) is not None else 1
                new_rem = int(m.group(4)) if m.group(4) is not None else 1
                hunk = {"ctx": m.group(5).strip(), "old": [], "new": [],
                        "old_changed": set(), "new_changed": set()}
                cur["hunks"].append(hunk)
    for f in files:
        f.pop("_plus", None)
    return files


# --------------------------------------------------------------------------- symbol tables
@dataclass
class Param:
    name: str
    default: str | None
    line: int
    lines: frozenset


@dataclass
class FuncSym:
    qualname: str
    params: list
    line: int
    sig_lines: frozenset
    body: list


@dataclass
class ClassSym:
    qualname: str
    line: int
    sig_lines: frozenset
    methods: frozenset


@dataclass
class FlagSym:
    owner: str
    key: str
    flags: tuple
    attrs: dict
    help: str | None
    line: int
    lines: frozenset


def _params(args: ast.arguments, lineno_of) -> list:
    def mk(a, d, prefix=""):
        end = (d.end_lineno if d is not None else a.end_lineno) or a.lineno
        lines = frozenset(lineno_of(l) for l in range(a.lineno, end + 1))
        return Param(prefix + a.arg, None if d is None else ast.unparse(d), lineno_of(a.lineno), lines)

    pos = args.posonlyargs + args.args
    dfl = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
    out = [mk(a, d) for a, d in zip(pos, dfl)]
    if args.vararg:
        out.append(mk(args.vararg, None, "*"))
    out += [mk(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        out.append(mk(args.kwarg, None, "**"))
    return [p for p in out if p.name not in ("self", "cls")]


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class _Collector(ast.NodeVisitor):
    """Collects functions, classes and CLI flags. Several trees may be fed (fragment mode)."""

    def __init__(self):
        self.funcs: dict[str, FuncSym] = {}
        self.classes: dict[str, ClassSym] = {}
        self.flags: dict[tuple, FlagSym] = {}
        self.scope: list[str] = []
        self.lineno_of = lambda l: l

    def size(self) -> int:
        return len(self.funcs) + len(self.classes) + len(self.flags)

    def feed(self, tree: ast.AST, lineno_of=lambda l: l):
        self.lineno_of, self.scope = lineno_of, []
        self.visit(tree)

    def _qual(self, name: str) -> str:
        return ".".join(self.scope + [name])

    def _sig_lines(self, node) -> frozenset:
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        stop = max(node.body[0].lineno, node.lineno + 1)
        return frozenset(self.lineno_of(l) for l in range(first, stop))

    def visit_ClassDef(self, node: ast.ClassDef):
        q = self._qual(node.name)
        methods = frozenset(n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        self.classes.setdefault(q, ClassSym(q, self.lineno_of(node.lineno), self._sig_lines(node), methods))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        q = self._qual(node.name)
        self.funcs.setdefault(q, FuncSym(
            q, _params(node.args, self.lineno_of), self.lineno_of(node.lineno),
            self._sig_lines(node), [ast.dump(s) for s in node.body]))
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and _call_name(dec.func) == "option":
                self._add_flag(dec, "click", q)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            self._add_flag(node, "argparse", ".".join(self.scope) or "<module>")
        self.generic_visit(node)

    def _add_flag(self, call: ast.Call, kind: str, owner: str):
        strs = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        flags, name = [], None
        for s in strs:
            if s.startswith("-"):
                flags.extend(p.strip() for p in s.split("/") if p.strip())
            elif name is None and s.isidentifier():
                name = s
        if not flags:
            return
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        if kind == "argparse" and "dest" in kw:
            try:
                name = str(ast.literal_eval(kw["dest"]))
            except Exception:
                pass
        if name is None:
            longs = [f for f in flags if f.startswith("--")]
            name = (longs or flags)[0].lstrip("-").replace("-", "_")
        names = _FLAG_ATTRS_CLICK if kind == "click" else _FLAG_ATTRS_ARGPARSE
        attrs = {a: ast.unparse(kw[a]) for a in names if a in kw}
        help_ = None
        if "help" in kw and isinstance(kw["help"], ast.Constant):
            help_ = str(kw["help"].value)
        key, n = (owner, name), 1
        while key in self.flags:
            n += 1
            key = (owner, f"{name}#{n}")
        end = call.end_lineno or call.lineno
        self.flags[key] = FlagSym(owner, key[1], tuple(flags), attrs, help_, self.lineno_of(call.lineno),
                                  frozenset(self.lineno_of(l) for l in range(call.lineno, end + 1)))


def _table_from_source(src: str | None) -> _Collector:
    col = _Collector()
    if src:
        try:
            col.feed(ast.parse(src))
        except SyntaxError as e:
            print(f"code_diff_parser: skipping unparsable file ({e})", file=sys.stderr)
    return col


def _lenient_parse(numbered: list):
    """Largest parsable slice of hunk lines (trim from start, then end). -> (tree, lineno_of) | (None, None)"""
    n = len(numbered)
    for start in range(min(n, _MAX_TRIM)):
        for end in range(n, max(start, n - _MAX_TRIM), -1):
            src = textwrap.dedent("\n".join(t for _, t in numbered[start:end])) + "\n"
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            nums = [ln for ln, _ in numbered[start:end]]
            return tree, (lambda l, nums=nums: nums[max(0, min(len(nums) - 1, l - 1))])
    return None, None


def _param_fragment(numbered: list) -> list:
    """Parse hunk lines as the inside of a parameter list ('name: T = default,' lines)."""
    n = len(numbered)
    for start in range(min(n, _MAX_TRIM)):
        for end in range(n, max(start, n - _MAX_TRIM), -1):
            body = textwrap.dedent("\n".join(t for _, t in numbered[start:end]))
            try:
                fn = ast.parse("def _f(\n" + body + "\n): pass\n").body[0]
            except SyntaxError:
                continue
            nums = [ln for ln, _ in numbered[start:end]]
            return _params(fn.args, lambda l, nums=nums: nums[max(0, min(len(nums) - 1, l - 2))])
    return []


# --------------------------------------------------------------------------- diffing tables
def _ev(symbol, change, old, new, file, line) -> dict:
    return {"symbol_name": symbol, "change_type": change, "old_value": old,
            "new_value": new, "file": file, "line": line}


def _pair(removed, added, score, threshold, same_parent=True):
    """Greedy best-first one-to-one pairing. Items are (id, sym). Optionally same parent scope only."""
    def parent(q): return q.rsplit(".", 1)[0] if "." in q else ""
    cands = []
    for r in removed:
        for a in added:
            if same_parent and parent(r[0]) != parent(a[0]):
                continue
            sc = score(r[1], a[1])
            if sc >= threshold:
                cands.append((sc, r, a))
    cands.sort(key=lambda t: -t[0])
    used_r, used_a, out = set(), set(), []
    for _, r, a in cands:
        if r[0] in used_r or a[0] in used_a:
            continue
        used_r.add(r[0]); used_a.add(a[0]); out.append((r, a))
    return out


def _func_score(o: FuncSym, n: FuncSym) -> float:
    body = difflib.SequenceMatcher(None, o.body, n.body, autojunk=False).ratio()
    names = difflib.SequenceMatcher(None, [p.name for p in o.params], [p.name for p in n.params],
                                    autojunk=False).ratio()
    return 0.6 * body + 0.4 * names


def _class_score(o: ClassSym, n: ClassSym) -> float:
    union = o.methods | n.methods
    return len(o.methods & n.methods) / len(union) if union else 0.0


def _diff_params(qual, oldp, newp, file, oc=None, nc=None) -> list:
    """Param rename/add/remove/default changes. oc/nc: changed-line filters (fragment mode)."""
    ev = []
    sm = difflib.SequenceMatcher(None, [p.name for p in oldp], [p.name for p in newp], autojunk=False)
    matched = []

    def touched_old(p): return oc is None or bool(p.lines & oc)
    def touched_new(p): return nc is None or bool(p.lines & nc)

    def removed(a):
        if touched_old(a):
            ev.append(_ev(f"{qual}.{a.name}", "param_removed", a.name, None, file, a.line))

    def added(b):
        if touched_new(b):
            ev.append(_ev(f"{qual}.{b.name}", "param_added", None, b.name, file, b.line))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += list(zip(oldp[i1:i2], newp[j1:j2]))
        elif tag == "replace":
            k = min(i2 - i1, j2 - j1)
            for a, b in zip(oldp[i1:i1 + k], newp[j1:j1 + k]):
                if touched_old(a) or touched_new(b):
                    ev.append(_ev(f"{qual}.{a.name}", "param_renamed", a.name, b.name, file, b.line))
                matched.append((a, b))
            for a in oldp[i1 + k:i2]:
                removed(a)
            for b in newp[j1 + k:j2]:
                added(b)
        elif tag == "delete":
            for a in oldp[i1:i2]:
                removed(a)
        elif tag == "insert":
            for b in newp[j1:j2]:
                added(b)
    for a, b in matched:
        if a.default != b.default and (touched_old(a) or touched_new(b)):
            ev.append(_ev(f"{qual}.{b.name}", "default_changed", a.default, b.default, file, b.line))
    return ev


def _flag_primary(f: FlagSym) -> str:
    return next((x for x in f.flags if x.startswith("--")), f.flags[0])


def _compare_flags(o: FlagSym, n: FlagSym, file) -> list:
    ev, of, nf = [], set(o.flags), set(n.flags)
    if of != nf:
        kind = ("flag_alias_added" if of < nf else "flag_alias_removed" if nf < of else "flag_renamed")
        ev.append(_ev(_flag_primary(o if kind == "flag_renamed" else n), kind,
                      ", ".join(o.flags), ", ".join(n.flags), file, n.line))
    for attr in sorted(set(o.attrs) | set(n.attrs)):
        if o.attrs.get(attr) != n.attrs.get(attr):
            ev.append(_ev(_flag_primary(n), f"flag_{attr}_changed", o.attrs.get(attr), n.attrs.get(attr),
                          file, n.line))
    return ev


def _diff_tables(old: _Collector, new: _Collector, file: str, oc=None, nc=None) -> list:
    ev, strict = [], oc is not None

    # classes
    rem = [(q, c) for q, c in old.classes.items() if q not in new.classes and (not strict or c.sig_lines & oc)]
    add = [(q, c) for q, c in new.classes.items() if q not in old.classes and (not strict or c.sig_lines & nc)]
    paired = _pair(rem, add, _class_score, 0.6)
    pr, pa = {r[0] for r, _ in paired}, {a[0] for _, a in paired}
    for r, a in paired:
        ev.append(_ev(r[0], "renamed", r[0], a[0], file, a[1].line))
    ev += [_ev(q, "removed", f"class {q}", None, file, c.line) for q, c in rem if q not in pr]
    ev += [_ev(q, "added", None, f"class {q}", file, c.line) for q, c in add if q not in pa]

    # functions
    rem = [(q, f) for q, f in old.funcs.items() if q not in new.funcs and (not strict or f.sig_lines & oc)]
    add = [(q, f) for q, f in new.funcs.items() if q not in old.funcs and (not strict or f.sig_lines & nc)]
    paired = _pair(rem, add, _func_score, 0.7)
    pr, pa = {r[0] for r, _ in paired}, {a[0] for _, a in paired}

    def sig(f: FuncSym):
        return f"{f.qualname}({', '.join(p.name + ('=' + p.default if p.default is not None else '') for p in f.params)})"

    for r, a in paired:
        ev.append(_ev(r[0], "renamed", r[0], a[0], file, a[1].line))
        ev += _diff_params(a[0], r[1].params, a[1].params, file, oc, nc)
    ev += [_ev(q, "removed", sig(f), None, file, f.line) for q, f in rem if q not in pr]
    ev += [_ev(q, "added", None, sig(f), file, f.line) for q, f in add if q not in pa]
    for q in old.funcs.keys() & new.funcs.keys():
        ev += _diff_params(q, old.funcs[q].params, new.funcs[q].params, file, oc, nc)

    # CLI flags
    def alive(f, ch): return not strict or bool(f.lines & ch)
    for k in old.flags.keys() & new.flags.keys():
        ev += _compare_flags(old.flags[k], new.flags[k], file)
    rem = [(str(i), f) for i, f in enumerate(f for k, f in old.flags.items() if k not in new.flags and alive(f, oc))]
    add = [(str(i), f) for i, f in enumerate(f for k, f in new.flags.items() if k not in old.flags and alive(f, nc))]

    def flag_score(o, n):
        if o.owner != n.owner:
            return 0.0
        if o.help and o.help == n.help:
            return 1.0
        only = (sum(1 for _, x in rem if x.owner == o.owner) == 1
                and sum(1 for _, x in add if x.owner == n.owner) == 1)
        same = (o.attrs.get("default") == n.attrs.get("default") and o.attrs.get("type") == n.attrs.get("type"))
        return 0.75 if only and same else 0.0

    paired = _pair(rem, add, flag_score, 0.75, same_parent=False)
    used_r = {r[0] for r, _ in paired}
    used_a = {a[0] for _, a in paired}
    for r, a in paired:
        ev += _compare_flags(r[1], a[1], file)
    ev += [_ev(_flag_primary(f), "flag_removed", ", ".join(f.flags), None, file, f.line)
           for i, f in rem if i not in used_r]
    ev += [_ev(_flag_primary(f), "flag_added", None, ", ".join(f.flags), file, f.line)
           for i, f in add if i not in used_a]
    return ev


# --------------------------------------------------------------------------- driver
def _fragment_events(path: str, hunks: list) -> list:
    old, new = _Collector(), _Collector()
    oc, nc, extra = set(), set(), []
    for h in hunks:
        oc |= h["old_changed"]; nc |= h["new_changed"]
        gained = []
        for col, side in ((old, h["old"]), (new, h["new"])):
            before = col.size()
            tree, lineno_of = _lenient_parse(side)
            if tree is not None:
                col.feed(tree, lineno_of)
            gained.append(col.size() - before)
        changed_txt = [t.strip() for ln, t in h["old"] if ln in h["old_changed"]] + \
                      [t.strip() for ln, t in h["new"] if ln in h["new_changed"]]
        looks_like_params = changed_txt and all(t.endswith(",") for t in changed_txt if t)
        if not any(gained) and looks_like_params:
            m = re.match(r"(?:class|def)\s+(\w+)", h["ctx"])
            owner = m.group(1) if m else "<unknown>"
            extra += _diff_params(owner, _param_fragment(h["old"]), _param_fragment(h["new"]), path,
                                  h["old_changed"], h["new_changed"])
    return _diff_tables(old, new, path, oc, nc) + extra


def extract_events(diff_text: str, old_sources: dict | None = None, new_sources: dict | None = None) -> list:
    """Main entry point. old_sources/new_sources: {path: full file text} enable exact FULL mode."""
    events, seen = [], set()
    for f in parse_unified_diff(diff_text):
        op, np_ = f["old_path"], f["new_path"]
        path = np_ or op
        if not path or not path.endswith(".py"):
            continue
        full = (old_sources is not None and new_sources is not None
                and (op is None or op in old_sources) and (np_ is None or np_ in new_sources))
        if full:
            evs = _diff_tables(_table_from_source(old_sources.get(op) if op else None),
                               _table_from_source(new_sources.get(np_) if np_ else None), path)
        else:
            evs = _fragment_events(path, f["hunks"])
        for e in evs:
            key = tuple(e.values())
            if key not in seen:
                seen.add(key); events.append(e)
    return events


def sources_from_git(repo: str, base: str, head: str, diff_text: str) -> tuple[dict, dict]:
    """Read full old/new contents of the .py files in the diff via `git show`."""
    def show(rev, path):
        r = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    old, new = {}, {}
    for f in parse_unified_diff(diff_text):
        if f["old_path"] and f["old_path"].endswith(".py"):
            s = show(base, f["old_path"])
            if s is not None:
                old[f["old_path"]] = s
        if f["new_path"] and f["new_path"].endswith(".py"):
            s = show(head, f["new_path"])
            if s is not None:
                new[f["new_path"]] = s
    return old, new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract symbol-change events from a unified diff of Python files.")
    ap.add_argument("diff", help="diff file, or - for stdin")
    ap.add_argument("--repo", help="git repo to read full old/new files from (enables exact FULL mode)")
    ap.add_argument("--head", help="head revision (with --repo)")
    ap.add_argument("--base", help="base revision (with --repo); default: <head>^")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.diff == "-" else open(a.diff, encoding="utf-8", errors="replace").read()
    old = new = None
    if a.repo:
        if not a.head:
            ap.error("--repo requires --head")
        old, new = sources_from_git(a.repo, a.base or a.head + "^", a.head, text)
    json.dump(extract_events(text, old, new), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
