#!/usr/bin/env python3
"""Tests for code_diff_parser.py and doc_parser.py.

Part A (SmokeTests)   - tiny SYNTHETIC inputs, only to prove the code runs and the logic is wired right.
Part B (RealExample)  - runs on the REAL Click commit + docs/options.md. Skipped unless you provide them:

    git clone https://github.com/pallets/click && cd click
    git show 1c20dc6e724c -- src/click/core.py > /tmp/click_1c20dc6e724c.diff
    cd ..
    DOCDRIFT_DIFF=/tmp/click_1c20dc6e724c.diff \\
    DOCDRIFT_DOC=click/docs/options.md \\
    DOCDRIFT_REPO=click DOCDRIFT_HEAD=1c20dc6e724c \\
    python test_parsers.py -v

DOCDRIFT_REPO/DOCDRIFT_HEAD are optional; with them the diff is analysed in exact FULL mode, without them
in FRAGMENT mode. The real test prints every event, and every doc unit that mentions a changed symbol.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import code_diff_parser as cdp
import doc_parser as dp

EVENT_KEYS = {"symbol_name", "change_type", "old_value", "new_value", "file", "line"}
UNIT_KEYS = {"text", "file", "line_range", "mentioned_symbols"}


def check_event(t, e):
    t.assertEqual(set(e), EVENT_KEYS)
    t.assertIsInstance(e["symbol_name"], str); t.assertTrue(e["symbol_name"])
    t.assertIsInstance(e["change_type"], str)
    for k in ("old_value", "new_value"):
        t.assertTrue(e[k] is None or isinstance(e[k], str))
    t.assertTrue(e["file"].endswith(".py"))
    t.assertIsInstance(e["line"], int); t.assertGreater(e["line"], 0)


def check_unit(t, u, nlines=None):
    t.assertEqual(set(u), UNIT_KEYS)
    t.assertTrue(u["text"].strip())
    a, b = u["line_range"]
    t.assertTrue(1 <= a <= b)
    if nlines:
        t.assertLessEqual(b, nlines)
    for s in u["mentioned_symbols"]:
        t.assertIn(s, u["text"])      # nothing fabricated: every symbol comes from the unit's own text


OLD = '''import argparse
import click


def old_name(a, b=1, timeout=30):
    total = a + b
    return total * timeout


class Gone:
    pass


@click.command()
@click.option("--retry-count", default=3, help="How many retries.")
@click.option("--verbose", "-v", is_flag=True)
def cli(retry_count, verbose):
    pass


def build(p):
    p.add_argument("--out", default="a.txt", help="Output file")
'''
NEW = OLD.replace("old_name(a, b=1, timeout=30)", "new_name(a, b=1, timeout=60)") \
         .replace("class Gone:\n    pass\n\n\n", "") \
         .replace('"--retry-count", default=3', '"--retries", default=5') \
         .replace("def cli(retry_count,", "def cli(retries,") \
         .replace('default="a.txt"', 'default="b.txt"')


@unittest.skipUnless(shutil.which("git"), "git not available")
class SmokeTests(unittest.TestCase):
    """SYNTHETIC data. Proves the scripts run; says nothing about Click."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        run = lambda *a: subprocess.run(["git", "-C", cls.d, *a], check=True, capture_output=True, text=True)
        run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
        for name, src in (("one", OLD), ("two", NEW)):
            with open(os.path.join(cls.d, "m.py"), "w") as fh:
                fh.write(src)
            run("add", "."); run("commit", "-q", "-m", name)
        cls.diff = run("diff", "HEAD~1", "HEAD").stdout

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def kinds(self, events):
        return {(e["change_type"], e["symbol_name"]) for e in events}

    def assert_expected(self, events):
        for e in events:
            check_event(self, e)
        k = self.kinds(events)
        self.assertIn(("renamed", "old_name"), k)
        self.assertIn(("default_changed", "new_name.timeout"), k)
        self.assertIn(("removed", "Gone"), k)
        self.assertIn(("flag_renamed", "--retry-count"), k)
        self.assertIn(("flag_default_changed", "--retries"), k)
        self.assertIn(("flag_default_changed", "--out"), k)
        self.assertIn(("param_renamed", "cli.retry_count"), k)

    def test_full_mode(self):
        old, new = cdp.sources_from_git(self.d, "HEAD~1", "HEAD", self.diff)
        self.assert_expected(cdp.extract_events(self.diff, old, new))

    def test_fragment_mode(self):
        self.assert_expected(cdp.extract_events(self.diff))

    def test_fragment_param_fallback(self):
        diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                "@@ -10,5 +10,5 @@ class Option(Parameter):\n"
                "         name: str,\n"
                "-        default: t.Any | None = None,\n"
                "+        default: t.Any | None = UNSET,\n"
                "         help: str | None = None,\n")
        ev = cdp.extract_events(diff)
        self.assertEqual([(e["symbol_name"], e["change_type"], e["old_value"], e["new_value"]) for e in ev],
                         [("Option.default", "default_changed", "None", "UNSET")])

    def test_non_python_ignored(self):
        self.assertEqual(cdp.extract_events("diff --git a/a.md b/a.md\n--- a/a.md\n+++ b/a.md\n@@ -1 +1 @@\n-a\n+b\n"), [])

    MD = """# Options

Use `@click.option` to add options. The `--name` flag is optional. See e.g. the docs. Defaults come from `default`.

- Set `show_default=True` to show it, or use `ParamType`.
- Env var `MY_APP_TOKEN` is read.

| Key | Default |
|-----|---------|
| `timeout` | `30` |

```python
@click.option("--skipped")
```
"""
    RST = """Title
=====

Some text about ``--verbose`` and :func:`click.echo`. Another sentence here.

.. note::

   Mind the ``timeout`` value.

.. option:: --retries N

   Retry count.

* Item with `default`::

      literal_block_skipped()

==== =======
Key  Default
==== =======
a    1
==== =======
"""

    def test_markdown(self):
        units = dp.parse_text(self.MD, "options.md")
        for u in units:
            check_unit(self, u, len(self.MD.splitlines()))
        texts = [u["text"] for u in units]
        self.assertIn("Options", texts)
        self.assertIn("See e.g. the docs.", texts)            # abbreviation did not split
        self.assertFalse(any("skipped" in t for t in texts))  # code fence skipped
        row = next(u for u in units if "timeout" in u["text"])
        self.assertEqual(row["line_range"], [10, 10])
        self.assertIn("timeout", row["mentioned_symbols"])
        flag = next(u for u in units if "--name" in u["text"])
        self.assertIn("--name", flag["mentioned_symbols"])
        item = next(u for u in units if "MY_APP_TOKEN" in u["text"])
        self.assertIn("MY_APP_TOKEN", item["mentioned_symbols"])
        self.assertIn("show_default", next(u for u in units if "show_default" in u["text"])["mentioned_symbols"])
        self.assertEqual(len(dp.parse_text(self.MD, "o.md", include_code=True)) - len(units), 1)

    def test_rst(self):
        units = dp.parse_text(self.RST, "options.rst")
        for u in units:
            check_unit(self, u, len(self.RST.splitlines()))
        texts = [u["text"] for u in units]
        self.assertIn("Title", texts)
        self.assertFalse(any("literal_block_skipped" in t for t in texts))
        self.assertIn("Mind the ``timeout`` value.", texts)
        opt = next(u for u in units if u["text"].startswith(".. option::"))
        self.assertIn("--retries", opt["mentioned_symbols"])
        echo = next(u for u in units if "click.echo" in u["text"])
        self.assertIn("click.echo", echo["mentioned_symbols"])
        self.assertTrue(any(t.startswith("a | 1") or t.startswith("a") and "1" in t for t in texts))


def _env():
    return os.environ.get("DOCDRIFT_DIFF"), os.environ.get("DOCDRIFT_DOC")


@unittest.skipUnless(all(_env()), "set DOCDRIFT_DIFF and DOCDRIFT_DOC to run on the real Click example")
class RealExample(unittest.TestCase):
    def test_real_click_example(self):
        diff_path, doc_path = _env()
        diff = open(diff_path, encoding="utf-8", errors="replace").read()
        repo, head = os.environ.get("DOCDRIFT_REPO"), os.environ.get("DOCDRIFT_HEAD")
        old = new = None
        if repo and head:
            old, new = cdp.sources_from_git(repo, os.environ.get("DOCDRIFT_BASE", head + "^"), head, diff)
        events = cdp.extract_events(diff, old, new)
        text = open(doc_path, encoding="utf-8", errors="replace").read()
        units = dp.parse_text(text, os.path.relpath(doc_path), fmt=None)

        print(f"\n=== mode: {'FULL (git show)' if old is not None else 'FRAGMENT (diff only)'} ===")
        print(f"=== {len(events)} symbol-change events ===")
        for e in events:
            check_event(self, e)
            print(e)
        print(f"=== {len(units)} doc units from {doc_path} ===")
        self.assertGreater(len(units), 0)
        for u in units:
            check_unit(self, u, len(text.splitlines()))

        print("=== doc units mentioning a changed symbol (bare name / last segment / flag) ===")
        for e in events:
            names = {e["symbol_name"], e["symbol_name"].rsplit(".", 1)[-1]}
            names |= {v for v in (e["old_value"], e["new_value"]) if v and len(v) < 40}
            hits = [u for u in units if names & set(u["mentioned_symbols"])]
            print(f"- {e['change_type']} {e['symbol_name']}: {len(hits)} unit(s)")
            for u in hits[:5]:
                print(f"    {u['file']}:{u['line_range']}  {u['text'][:90]!r}")


if __name__ == "__main__":
    unittest.main()
