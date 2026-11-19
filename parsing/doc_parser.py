#!/usr/bin/env python3
"""DocDrift doc_parser: Markdown or reStructuredText -> doc units.

Unit schema (exactly these four keys):
    {text, file, line_range, mentioned_symbols}

* line_range is [first_line, last_line], 1-based, inclusive.
* One unit per: heading, table row, list item (incl. reST field-list entry), or sentence of a paragraph.
* Fenced / literal code blocks are skipped unless include_code=True (then one unit per block).
* mentioned_symbols: names from code spans (`x`, ``x``, :role:`x`), --flags / -f, and identifier-looking
  plain tokens (snake_case, CamelCase, dotted.names, ALLCAPS, name()). Dotted names also yield their last
  segment. Every symbol is a substring of the unit's text.
"""
from __future__ import annotations

import argparse
import bisect
import json
import keyword
import os
import re
import sys

# --------------------------------------------------------------------------- symbol extraction
_CODE_RE = re.compile(r"(`+)(.+?)\1")
_URL_RE = re.compile(r"https?://[^\s<>)\]`]+")
_LINK_URL_RE = re.compile(r"\]\([^)]*\)")
_FLAG_RE = re.compile(r"(?<![\w/.\-])--?[A-Za-z][\w\-]*")
_IDENT_FULL = re.compile(r"[~.!]?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?")
_IDENT_ANY = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_PLAIN = [
    re.compile(r"\b__\w+__\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"),
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),
    re.compile(r"\b[A-Z][A-Z0-9]{3,}\b"),
    re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b"),
    re.compile(r"\b([A-Za-z_]\w*)\(\)"),
]
_EXT = {"py", "md", "rst", "txt", "toml", "cfg", "ini", "json", "yaml", "yml", "html", "js", "ts", "css",
        "lock", "com", "org", "io", "net", "dev", "pyi", "sh", "png", "svg", "xml", "csv"}
_NOT_SYMBOLS = set(keyword.kwlist) | {"None", "True", "False"}
_CAPS_STOP = {"NOTE", "WARNING", "TODO", "FIXME", "JSON", "HTTP", "HTTPS", "HTML", "YAML", "TOML", "ASCII",
              "POSIX", "UNIX", "LINUX", "MACOS", "XML", "CSV", "TRUE", "FALSE", "NONE", "IMPORTANT", "TIP",
              "HINT", "DEPRECATED", "SHOULD", "ALSO", "WITH", "FROM", "THIS", "THAT", "MUST", "ONLY"}


def _symbols_from_code(c: str) -> list:
    c = _URL_RE.sub(" ", c).strip()
    out = []
    if _IDENT_FULL.fullmatch(c):
        name = c.lstrip("~.!").removesuffix("()")
        out.append(name)
        if "." in name:
            last = name.rsplit(".", 1)[1]
            if last not in _EXT:
                out.append(last)
        return out
    out += [m.group(0).rstrip("-") for m in _FLAG_RE.finditer(c)]
    for m in _IDENT_ANY.finditer(c):
        name = m.group(0)
        out.append(name)
        if "." in name:
            last = name.rsplit(".", 1)[1]
            if last not in _EXT:
                out.append(last)
    return out


def extract_symbols(text: str) -> list:
    """Ordered, de-duplicated symbol names mentioned in text."""
    found: list[tuple[int, str]] = []
    for m in _CODE_RE.finditer(text):
        for s in _symbols_from_code(m.group(2).strip()):
            found.append((m.start(), s))
    rest = _CODE_RE.sub(lambda m: " " * len(m.group(0)), text)
    rest = _LINK_URL_RE.sub(lambda m: " " * len(m.group(0)), rest)
    rest = _URL_RE.sub(lambda m: " " * len(m.group(0)), rest)
    for m in _FLAG_RE.finditer(rest):
        found.append((m.start(), m.group(0).rstrip("-")))
    for pat in _PLAIN:
        for m in pat.finditer(rest):
            tok = m.group(1) if pat.groups else m.group(0)
            if pat is _PLAIN[5] and tok in _CAPS_STOP:
                continue
            if "." in tok:
                if len(tok.split(".")[0]) < 2 or tok.rsplit(".", 1)[1] in _EXT:
                    continue
                found.append((m.start(), tok))
                found.append((m.start() + len(tok), tok.rsplit(".", 1)[1]))
            else:
                found.append((m.start(), tok))
    seen, out = set(), []
    for _, s in sorted(found, key=lambda t: t[0]):
        s = s.strip()
        if len(s) < 2 or s in _NOT_SYMBOLS or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# --------------------------------------------------------------------------- sentence splitting
_ABBR = {"e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "mr.", "dr.", "no.", "eg.", "ie."}
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]*_]*\s+(?=[\"'(\[*_]*[A-Z0-9\x00])")


def _mask(s: str) -> str:
    z = lambda m: "\x00" * len(m.group(0))
    return _URL_RE.sub(z, _LINK_URL_RE.sub(z, _CODE_RE.sub(z, s)))


def _unit(text, file, a, b) -> dict:
    return {"text": text, "file": file, "line_range": [a, b], "mentioned_symbols": extract_symbols(text)}


def _sentence_units(para: list, file: str) -> list:
    """para: [(lineno, text)] of one paragraph -> one unit per sentence."""
    starts, parts, pos = [], [], 0
    for ln, t in para:
        starts.append(pos); parts.append(t); pos += len(t) + 1
    joined, masked = " ".join(parts), _mask(" ".join(parts))
    cuts = [0]
    for m in _BOUNDARY.finditer(masked):
        words = masked[:m.start()].split()
        if words and words[-1].lower() in _ABBR:
            continue
        if words and re.fullmatch(r"[A-Z]\.", words[-1]):
            continue
        cuts.append(m.end())
    cuts.append(len(joined) + 1)
    units = []
    for a, b in zip(cuts, cuts[1:]):
        seg = joined[a:b].strip()
        if not seg:
            continue
        s_off = a + (len(joined[a:b]) - len(joined[a:b].lstrip()))
        e_off = s_off + len(seg) - 1
        la = para[bisect.bisect_right(starts, s_off) - 1][0]
        lb = para[bisect.bisect_right(starts, e_off) - 1][0]
        units.append(_unit(seg, file, la, lb))
    return units


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


# --------------------------------------------------------------------------- Markdown
_MD_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_MD_HEAD = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_MD_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(\S.*)$")
_MD_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_MD_REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
_MD_TAGLINE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$")


def parse_markdown(text: str, file: str, include_code: bool = False) -> list:
    lines = text.splitlines()
    n, i, units, para = len(lines), 0, [], []
    in_table = False

    def flush():
        if para:
            units.extend(_sentence_units(list(para), file))
            para.clear()

    if lines and lines[0].strip() == "---":
        for j in range(1, n):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    while i < n:
        raw = re.sub(r"^\s*(>\s?)+", "", lines[i]) if lines[i].lstrip().startswith(">") else lines[i]
        s, ln = raw.strip(), i + 1
        m = _MD_FENCE.match(raw)
        if m:
            flush(); in_table = False
            ch, k = m.group(1)[0], len(m.group(1))
            j = i + 1
            while j < n:
                t = lines[j].strip()
                if t and set(t) == {ch} and len(t) >= k:
                    break
                j += 1
            body = lines[i + 1:j]
            if include_code and any(b.strip() for b in body):
                units.append(_unit("\n".join(body), file, i + 2, max(i + 2, j)))
            i = j + 1
            continue
        if not s:
            flush(); in_table = False; i += 1; continue
        if s.startswith(":::") or _MD_REFDEF.match(raw) or _MD_TAGLINE.match(raw):
            flush(); i += 1; continue
        if s.startswith("<!--"):
            flush()
            while i < n and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        h = _MD_HEAD.match(raw)
        if h:
            flush(); in_table = False
            units.append(_unit(h.group(1), file, ln, ln)); i += 1; continue
        if para and re.fullmatch(r"=+|-+", s):          # setext heading
            a = para[0][0]
            units.append(_unit(" ".join(t for _, t in para), file, a, para[-1][0]))
            para.clear(); i += 1; continue
        if _MD_HR.match(raw):
            flush(); i += 1; continue
        if "|" in s and _MD_TABLE_SEP.match(s) and ("-" in s) and (in_table or s.startswith("|")):
            in_table = True; i += 1; continue
        is_row = s.startswith("|") or (in_table and "|" in s) or (
            "|" in s and i + 1 < n and "|" in lines[i + 1] and _MD_TABLE_SEP.match(lines[i + 1].strip()))
        if is_row:
            flush(); in_table = True
            units.append(_unit(s, file, ln, ln)); i += 1; continue
        in_table = False
        it = _MD_ITEM.match(raw)
        if it:
            flush()
            parts, last = [it.group(3).strip()], i
            j = i + 1
            while j < n and lines[j].strip() and not _MD_ITEM.match(lines[j]) and not _MD_HEAD.match(lines[j]) \
                    and not _MD_FENCE.match(lines[j]) and not lines[j].strip().startswith(("|", ":::")):
                parts.append(lines[j].strip()); last = j; j += 1
            units.append(_unit(" ".join(parts), file, ln, last + 1))
            i = j
            continue
        para.append((ln, s)); i += 1
    flush()
    return units


# --------------------------------------------------------------------------- reStructuredText
_RST_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+(?:\|[^|]+\|\s+)?([\w:+.\-]+)::\s*(.*)$")
_RST_COMMENTISH = re.compile(r"^\s*\.\.(\s|$)")
_RST_UNDERLINE = re.compile(r"^([!-/:-@\[-`{-~])\1{2,}\s*$")
_RST_GRID_BORDER = re.compile(r"^\s*\+[-=+:]+\+\s*$")
_RST_GRID_ROW = re.compile(r"^\s*\|.*\|\s*$")
_RST_SIMPLE_BORDER = re.compile(r"^\s*=+(\s+=+)+\s*$")
_RST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)]|#[.)])\s+(\S.*)$")
_RST_FIELD = re.compile(r"^(\s*):[^:\s`][^:]*:(\s|$)")
_RST_CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "highlight", "literalinclude", "parsed-literal",
                        "testcode", "testoutput", "doctest", "ipython", "include", "image", "figure",
                        "toctree", "csv-table", "raw", "math", "productionlist", "autoclass", "automodule",
                        "autofunction", "automethod", "click"}
_RST_API_DIRECTIVES = {"option", "program", "function", "class", "method", "attribute", "data", "exception",
                       "module", "cmdoption", "envvar", "describe", "decorator", "staticmethod", "classmethod"}


def _skip_block(lines: list, i: int, base: int) -> int:
    """Index after the indented block that follows line i (lines more indented than `base`)."""
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > base):
        j += 1
    return j


def parse_rst(text: str, file: str, include_code: bool = False) -> list:
    return _parse_rst_lines(text.splitlines(), file, include_code, 0)


def _parse_rst_lines(lines: list, file: str, include_code: bool, off: int) -> list:
    n, i, units, para = len(lines), 0, [], []
    in_simple, borders = False, 0
    pending_literal = None

    def flush():
        nonlocal pending_literal
        if para:
            if para[-1][1].endswith("::"):
                pending_literal = para_indent[0]
            units.extend(_sentence_units(list(para), file))
            para.clear()

    para_indent = [0]

    def emit_code(a, b):
        body = [l for l in lines[a:b]]
        while body and not body[-1].strip():
            body.pop()
        if include_code and any(x.strip() for x in body):
            import textwrap
            units.append(_unit(textwrap.dedent("\n".join(body)), file, off + a + 1, off + a + len(body)))

    while i < n:
        raw = lines[i].rstrip()
        s, ln = raw.strip(), off + i + 1
        if not s:
            flush()
            if in_simple and borders >= 2:
                in_simple, borders = False, 0
            if pending_literal is not None:
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and _indent(lines[j]) > pending_literal:
                    k = j
                    while k < n and (not lines[k].strip() or _indent(lines[k]) > pending_literal):
                        k += 1
                    emit_code(j, k); i = k
                else:
                    i += 1
                pending_literal = None
                continue
            i += 1
            continue
        pending_literal = None
        d = _RST_DIRECTIVE.match(raw)
        if d:
            flush()
            base, name, arg = len(d.group(1)), d.group(2), d.group(3).strip()
            end = _skip_block(lines, i, base)
            body_start = i + 1
            while body_start < end and re.match(r"^\s+:[\w\-]+:", lines[body_start]):
                body_start += 1                                   # directive options
            if name in _RST_CODE_DIRECTIVES:
                emit_code(i + 1, end)
            else:
                if name in _RST_API_DIRECTIVES or name.startswith(("py:", "c:", "cpp:", "std:")):
                    u = _unit(f".. {name}:: {arg}", file, ln, ln)
                    u["mentioned_symbols"] = list(dict.fromkeys(_symbols_from_code(arg) + u["mentioned_symbols"]))
                    u["mentioned_symbols"] = [x for x in u["mentioned_symbols"]
                                              if len(x) >= 2 and x not in _NOT_SYMBOLS and x in u["text"]]
                    units.append(u)
                elif arg and not re.fullmatch(r"[\d.]+\w*", arg):
                    units.extend(_sentence_units([(ln, arg)], file))
                block = [l for l in lines[body_start:end]]
                ind = min((_indent(l) for l in block if l.strip()), default=0)
                units.extend(_parse_rst_lines([l[ind:] if l.strip() else "" for l in block], file,
                                              include_code, off + body_start))
            i = end
            continue
        if _RST_COMMENTISH.match(raw):
            flush(); i = _skip_block(lines, i, _indent(raw)); continue
        if _RST_SIMPLE_BORDER.match(raw):
            flush()
            if not in_simple:
                in_simple, borders = True, 1
            else:
                borders += 1
                if borders >= 3:
                    in_simple, borders = False, 0
            i += 1; continue
        if in_simple:
            units.append(_unit(re.sub(r"\s{2,}", " | ", s), file, ln, ln)); i += 1; continue
        if _RST_GRID_BORDER.match(raw):
            flush(); i += 1; continue
        if _RST_GRID_ROW.match(raw):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if any(cells):
                units.append(_unit(s, file, ln, ln))
            i += 1; continue
        if not para and s.startswith(">>>"):
            while i < n and lines[i].strip():
                i += 1
            continue
        if not para and _RST_UNDERLINE.match(raw) and i + 2 < n and _RST_UNDERLINE.match(lines[i + 2].rstrip()):
            t = lines[i + 1].strip()                                  # overline + title + underline
            units.append(_unit(t, file, ln + 1, ln + 1)); i += 3; continue
        if not para and _RST_UNDERLINE.match(raw):
            i += 1; continue                                          # transition / stray underline
        if not para and i + 1 < n and _RST_UNDERLINE.match(lines[i + 1].rstrip()) \
                and len(lines[i + 1].strip()) >= len(s) and not _RST_ITEM.match(raw):
            units.append(_unit(s, file, ln, ln)); i += 2; continue    # section title
        it = _RST_ITEM.match(raw) or _RST_FIELD.match(raw)
        if it and not para:
            base = _indent(raw)
            parts, last, j = [re.sub(r"^\s*([-*+]|\d+[.)]|#[.)])\s+", "", raw).strip()], i, i + 1
            while j < n and lines[j].strip() and _indent(lines[j]) > base:
                parts.append(lines[j].strip()); last = j; j += 1
            units.append(_unit(" ".join(parts), file, ln, off + last + 1))
            if parts[-1].endswith("::"):
                pending_literal = base
            i = j
            continue
        if not para:
            para_indent[0] = _indent(raw)
        para.append((ln, s)); i += 1
    flush()
    return units


# --------------------------------------------------------------------------- entry points
def parse_text(text: str, file: str, fmt: str | None = None, include_code: bool = False) -> list:
    fmt = fmt or _detect(file)
    return (parse_markdown if fmt == "markdown" else parse_rst)(text, file, include_code)


def _detect(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown", ".mdown", ".mdx"):
        return "markdown"
    if ext in (".rst", ".rest", ".txt"):
        return "rst"
    raise ValueError(f"unsupported doc type: {path} (expected .md or .rst)")


def parse_file(path: str, display_name: str | None = None, include_code: bool = False) -> list:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_text(fh.read(), display_name or path, include_code=include_code)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Split a Markdown/reST file into doc units (JSON).")
    ap.add_argument("path")
    ap.add_argument("--name", help="value for the `file` field (default: the path as given)")
    ap.add_argument("--include-code", action="store_true", help="emit one unit per code block")
    a = ap.parse_args(argv)
    json.dump(parse_file(a.path, a.name, a.include_code), sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
