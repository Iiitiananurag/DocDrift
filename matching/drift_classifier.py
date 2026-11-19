"""
DocDrift Classifier (`drift_classifier.py`)

Classifies each matched `(doc_unit, symbol_event, score)` pair into:
  - "still correct"
  - "likely outdated"
  - "uncertain"
with a concise one-line reason and confidence score.

Supports:
  - Groq OpenAI-compatible API (`llama-3.1-8b-instant` by default) via `GROQ_API_KEY`
  - OpenRouter free-tier models (`meta-llama/llama-3.1-8b-instruct:free` or `openrouter/free`)
    via `OPENROUTER_API_KEY`
  - Deterministic local mock mode (`mock=True` or `DOCDRIFT_MOCK=1`) for fast, offline CI
    and local testing without an API call.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


VALID_VERDICTS = ("still correct", "likely outdated", "uncertain")

SYSTEM_PROMPT = (
    "You are DocDrift, a precise CI documentation verifier. "
    "Given a documentation snippet and a code change, classify whether the documentation is: "
    '"still correct", "likely outdated", or "uncertain". '
    'Respond ONLY with valid JSON: {"verdict": "<still correct|likely outdated|uncertain>", '
    '"reason": "<one-line explanation under 120 chars>", "confidence": <float 0.0 to 1.0>}.'
)


@dataclass
class DriftClassification:
    doc_unit: Dict[str, Any]
    symbol_event: Dict[str, Any]
    match_score: float
    verdict: str  # "still correct" | "likely outdated" | "uncertain"
    reason: str
    confidence: float
    provider: str  # e.g., "groq:llama-3.1-8b-instant", "openrouter:...", or "mock"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_short_prompt(doc_unit: Dict[str, Any], symbol_event: Dict[str, Any]) -> str:
    """
    Builds a compact, token-efficient prompt (<120 tokens) for high-volume CI runs.
    """
    doc_text = doc_unit.get("text", "").strip()
    sym = symbol_event.get("symbol_name", "")
    ctype = symbol_event.get("change_type", "modified")
    old_val = symbol_event.get("old_value", "")
    new_val = symbol_event.get("new_value", "")
    code_loc = f"{symbol_event.get('file', '')}:{symbol_event.get('line', 1)}"

    return (
        f"DOC ({doc_unit.get('file', 'docs')}:{doc_unit.get('line_range', [1, 1])[0]}):\n"
        f'"{doc_text}"\n\n'
        f"CODE CHANGE ({code_loc}):\n"
        f"symbol={sym} | type={ctype}\n"
        f"old: {old_val}\n"
        f"new: {new_val}\n\n"
        "Return JSON with keys: verdict, reason, confidence."
    )


# ---------------------------------------------------------------------------
# Deterministic Mock Classifier (for local testing & offline CI)
# ---------------------------------------------------------------------------

def _extract_literals(text: str) -> List[str]:
    """Extracts numbers, durations, quoted strings, flags, and parameter names."""
    literals: List[str] = []
    # Numeric values (ports, timeouts, retries, versions)
    for m in re.findall(r"\b\d+(?:\.\d+)?(?:s|ms|m|h|gb|mb|kb)?\b", text.lower()):
        literals.append(m)
    # Parameter / flag names like `--timeout` or `timeout=`
    for m in re.findall(r"--[a-zA-Z0-9_-]+|[a-zA-Z_][a-zA-Z0-9_]*(?=\s*=)", text):
        literals.append(m.strip())
    # Quoted string literals
    for m in re.findall(r"['\"]([^'\"]{2,40})['\"]", text):
        literals.append(m.strip())
    return literals


def _mock_classify_pair(
    doc_unit: Dict[str, Any],
    symbol_event: Dict[str, Any],
    match_score: float,
) -> Tuple[str, str, float]:
    """
    Rule-based local heuristic that inspects old vs. new code values against
    the documentation sentence when running in `--mock` mode.
    """
    doc_text = doc_unit.get("text", "")
    doc_lower = doc_text.lower()
    sym = symbol_event.get("symbol_name", "")
    ctype = symbol_event.get("change_type", "modified").lower()
    old_val = symbol_event.get("old_value", "")
    new_val = symbol_event.get("new_value", "")

    # 1. Removed or deprecated symbol still referenced in docs
    if "remove" in ctype or "delete" in ctype or (old_val and not new_val.strip()):
        return (
            "likely outdated",
            f"`{sym}` was removed in code (`{old_val}`), but documentation still instructs users to call or configure it.",
            round(min(0.97, 0.82 + 0.15 * match_score), 2),
        )

    # 2. Renamed symbol
    if "rename" in ctype:
        return (
            "likely outdated",
            f"`{sym}` was renamed from `{old_val}` to `{new_val}`, leaving the documented identifier stale.",
            round(min(0.96, 0.80 + 0.15 * match_score), 2),
        )

    # 3. Check if specific old literals (numbers, defaults, flags, endpoints) appear in the doc text
    old_literals = _extract_literals(old_val)
    new_literals = _extract_literals(new_val)
    removed_literals = [lit for lit in old_literals if lit not in new_literals]

    matched_old_literals = [
        lit for lit in removed_literals
        if re.search(rf"\b{re.escape(lit.lower())}\b", doc_lower) or lit.lower() in doc_lower
    ]
    matched_new_literals = [
        lit for lit in new_literals
        if lit not in old_literals and (re.search(rf"\b{re.escape(lit.lower())}\b", doc_lower) or lit.lower() in doc_lower)
    ]

    if matched_old_literals:
        stale_tok = matched_old_literals[0]
        replacement = next((n for n in new_literals if n not in old_literals), new_val)
        return (
            "likely outdated",
            f"Doc states `{stale_tok}` for `{sym}`, but code changed from `{old_val}` to `{new_val}` (now `{replacement}`).",
            round(min(0.95, 0.80 + 0.15 * match_score), 2),
        )

    if matched_new_literals:
        return (
            "still correct",
            f"Doc already reflects the updated value (`{matched_new_literals[0]}`) for `{sym}`.",
            0.90,
        )

    # 4. Signature or required-parameter change
    if "signature" in ctype or "param" in ctype or "arg" in ctype:
        if old_val != new_val:
            return (
                "likely outdated",
                f"Signature of `{sym}` changed from `{old_val}` to `{new_val}`; documented call example or parameters may break.",
                round(min(0.90, 0.72 + 0.18 * match_score), 2),
            )

    # 5. High-similarity semantic match on behavioral/default change without explicit literal match
    if "default" in ctype or "behavior" in ctype or "config" in ctype:
        if match_score >= 0.45 and old_val != new_val:
            return (
                "likely outdated",
                f"Behavior/default of `{sym}` changed (`{old_val}` -> `{new_val}`), which conflicts with documented behavior.",
                round(min(0.88, 0.65 + 0.25 * match_score), 2),
            )

    # 6. Internal refactor where doc only mentions high-level concept without stale details
    if "refactor" in ctype or "internal" in ctype or "type_hint" in ctype:
        return (
            "still correct",
            f"Code change to `{sym}` is an internal refactor (`{old_val}` -> `{new_val}`) that does not alter documented usage.",
            0.86,
        )

    if match_score >= 0.75:
        return (
            "uncertain",
            f"Doc mentions `{sym}` which changed (`{old_val}` -> `{new_val}`), but prose does not state the specific old value.",
            0.62,
        )

    return (
        "still correct",
        f"Doc unit does not depend on the modified details of `{sym}`.",
        0.78,
    )


# ---------------------------------------------------------------------------
# Free Open-Weight LLM API Caller (Groq / OpenRouter)
# ---------------------------------------------------------------------------

def _call_llm_api(
    prompt: str,
    api_key: str,
    provider: str = "groq",
    model: Optional[str] = None,
    timeout_sec: int = 20,
) -> Tuple[str, str, float, str]:
    """
    Calls Groq (`https://api.groq.com/openai/v1/chat/completions`) or OpenRouter
    (`https://openrouter.ai/api/v1/chat/completions`) with JSON response formatting.
    """
    provider_clean = provider.lower().strip()
    if provider_clean == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        chosen_model = model or os.environ.get("DOCDRIFT_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        chosen_model = model or os.environ.get("DOCDRIFT_MODEL", "llama-3.1-8b-instant")

    payload: Dict[str, Any] = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 600,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "DocDrift-CI/1.0",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    content = body["choices"][0]["message"]["content"].strip()
    # Extract JSON object even if wrapped in markdown code fences
    json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    parsed = json.loads(json_match.group(0) if json_match else content)

    verdict = str(parsed.get("verdict", "uncertain")).lower().strip()
    if verdict not in VALID_VERDICTS:
        verdict = "uncertain"
    reason = str(parsed.get("reason", "LLM evaluated symbol change against doc unit.")).replace("\n", " ").strip()
    confidence = float(parsed.get("confidence", 0.75))
    confidence = max(0.0, min(1.0, round(confidence, 2)))

    return verdict, reason, confidence, f"{provider_clean}:{chosen_model}"


# ---------------------------------------------------------------------------
# Public Classification Entry Point
# ---------------------------------------------------------------------------

def classify_matched_pairs(
    matched_pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any], float]],
    api_key: Optional[str] = None,
    provider: str = "groq",
    model: Optional[str] = None,
    mock: Optional[bool] = None,
) -> List[DriftClassification]:
    """
    Classifies each `(doc_unit, symbol_event, score)` tuple returned by `matcher.py`.

    API Key resolution order:
      1. Explicit `api_key` argument
      2. `GROQ_API_KEY` environment variable (sets provider="groq")
      3. `OPENROUTER_API_KEY` environment variable (sets provider="openrouter")
      4. `DOCDRIFT_API_KEY` environment variable

    Mock mode activation:
      - `mock=True` passed explicitly, OR
      - `DOCDRIFT_MOCK=1` / `true` set in environment, OR
      - No API key is set and `DOCDRIFT_AUTO_MOCK=1`.
    """
    env_mock = os.environ.get("DOCDRIFT_MOCK", "").lower() in ("1", "true", "yes")
    use_mock = bool(mock) if mock is not None else env_mock

    resolved_key = api_key
    resolved_provider = provider
    if not resolved_key:
        if os.environ.get("GROQ_API_KEY"):
            resolved_key = os.environ["GROQ_API_KEY"]
            resolved_provider = "groq"
        elif os.environ.get("OPENROUTER_API_KEY"):
            resolved_key = os.environ["OPENROUTER_API_KEY"]
            resolved_provider = "openrouter"
        elif os.environ.get("DOCDRIFT_API_KEY"):
            resolved_key = os.environ["DOCDRIFT_API_KEY"]

    if not use_mock and not resolved_key:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY (or OPENROUTER_API_KEY), "
            "or pass --mock / DOCDRIFT_MOCK=1 for local offline testing."
        )

    results: List[DriftClassification] = []
    for doc_unit, symbol_event, score in matched_pairs:
        if use_mock:
            verdict, reason, confidence = _mock_classify_pair(doc_unit, symbol_event, score)
            used_provider = "mock (local-deterministic)"
        else:
            prompt = build_short_prompt(doc_unit, symbol_event)
            try:
                verdict, reason, confidence, used_provider = _call_llm_api(
                    prompt=prompt,
                    api_key=resolved_key or "",
                    provider=resolved_provider,
                    model=model,
                )
            # except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            #     # Graceful fallback if network/rate-limit error occurs during CI
            #     verdict, reason, confidence = _mock_classify_pair(doc_unit, symbol_event, score)
            #     used_provider = f"fallback-mock (api_error: {type(exc).__name__})"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                print(f"!!! REAL API ERROR: {type(exc).__name__}: {exc}")
                if hasattr(exc, 'read'):
                   print(f"!!! BODY: {exc.read().decode()}")
                verdict, reason, confidence = _mock_classify_pair(doc_unit, symbol_event, score)
                used_provider = f"fallback-mock (api_error: {type(exc).__name__})"

        results.append(
            DriftClassification(
                doc_unit=doc_unit,
                symbol_event=symbol_event,
                match_score=score,
                verdict=verdict,
                reason=reason,
                confidence=confidence,
                provider=used_provider,
            )
        )

    return results