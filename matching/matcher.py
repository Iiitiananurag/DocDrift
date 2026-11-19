"""
DocDrift Matcher (`matcher.py`)

Matches code symbol-change events against checkable documentation units using:
1. Exact or fuzzy symbol-name matching against `mentioned_symbols`.
2. Semantic embedding fallback using `sentence-transformers` (`all-MiniLM-L6-v2`)
   comparing the doc unit's text with a natural-language description of the code
   change. Includes an automatic local word/n-gram cosine fallback when running in
   air-gapped environments without cached Hugging Face weights.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data Models (accept both dicts and dataclasses seamlessly)
# ---------------------------------------------------------------------------

@dataclass
class SymbolChangeEvent:
    symbol_name: str
    change_type: str  # e.g., "signature_changed", "default_changed", "removed", "renamed", "behavior_changed"
    old_value: str
    new_value: str
    file: str
    line: int

    @classmethod
    def from_obj(cls, obj: Any) -> "SymbolChangeEvent":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(
                symbol_name=str(obj.get("symbol_name", "")),
                change_type=str(obj.get("change_type", "modified")),
                old_value=str(obj.get("old_value", "")),
                new_value=str(obj.get("new_value", "")),
                file=str(obj.get("file", "unknown")),
                line=int(obj.get("line", 1)),
            )
        raise TypeError(f"Unsupported SymbolChangeEvent input: {type(obj)}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocUnit:
    text: str
    file: str
    line_range: Tuple[int, int]
    mentioned_symbols: List[str]

    @classmethod
    def from_obj(cls, obj: Any) -> "DocUnit":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            lr = obj.get("line_range", (1, 1))
            if isinstance(lr, (list, tuple)) and len(lr) >= 2:
                line_range = (int(lr[0]), int(lr[1]))
            elif isinstance(lr, int):
                line_range = (lr, lr)
            else:
                line_range = (1, 1)
            return cls(
                text=str(obj.get("text", "")),
                file=str(obj.get("file", "README.md")),
                line_range=line_range,
                mentioned_symbols=[str(s) for s in obj.get("mentioned_symbols", [])],
            )
        raise TypeError(f"Unsupported DocUnit input: {type(obj)}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["line_range"] = list(self.line_range)
        return d


# ---------------------------------------------------------------------------
# Natural-Language Change Description & Symbol Normalization
# ---------------------------------------------------------------------------

def _split_identifier(name: str) -> List[str]:
    """Splits camelCase, PascalCase, snake_case, and dotted names into lowercase tokens."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.]", " ", name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cleaned)
    tokens = re.split(r"[_\.\s]+", spaced.lower())
    return [t for t in tokens if t]


def _normalize_symbol(name: str) -> str:
    """Canonical lowercase alphanumeric representation for fuzzy comparison."""
    return "".join(_split_identifier(name))


def describe_symbol_event(event: Dict[str, Any] | SymbolChangeEvent) -> str:
    """
    Converts a code symbol-change event into a short natural-language description
    suitable for semantic embedding similarity against documentation prose.
    """
    ev = SymbolChangeEvent.from_obj(event)
    human_words = " ".join(_split_identifier(ev.symbol_name))
    ct = ev.change_type.lower().replace("_", " ")

    if "remove" in ct or "delete" in ct or not ev.new_value.strip():
        return (
            f"Function or setting {ev.symbol_name} ({human_words}) in {ev.file} "
            f"was removed or deprecated (previously {ev.old_value})."
        )
    if "rename" in ct:
        return (
            f"{ev.symbol_name} ({human_words}) in {ev.file} was renamed or moved "
            f"from {ev.old_value} to {ev.new_value}."
        )
    return (
        f"In {ev.file}, {ev.symbol_name} ({human_words}) had a {ct}: "
        f"changed from '{ev.old_value}' to '{ev.new_value}'."
    )


# ---------------------------------------------------------------------------
# Exact & Fuzzy Symbol Matching (`mentioned_symbols`)
# ---------------------------------------------------------------------------

def compute_symbol_match_score(
    symbol_name: str,
    mentioned_symbols: Iterable[str],
    fuzzy_threshold: float = 0.78,
) -> Tuple[float, str]:
    """
    Checks whether `symbol_name` matches any symbol in `mentioned_symbols`
    via exact or fuzzy identifier comparison.

    Returns:
        (score, match_method) where score is in [0.0, 1.0] and match_method is
        one of "exact_symbol", "fuzzy_symbol", or "none".
    """
    if not symbol_name:
        return 0.0, "none"

    candidates = [c for c in (mentioned_symbols or []) if str(c).strip()]
    if not candidates:
        return 0.0, "none"

    target_raw = symbol_name.strip().split("(")[0]
    target_lower = target_raw.lower()
    target_norm = _normalize_symbol(target_raw)
    target_leaf = target_raw.split(".")[-1].lower()

    best_score = 0.0
    best_method = "none"

    for cand in candidates:
        cand_raw = str(cand).strip().split("(")[0]
        if not cand_raw:
            continue
        cand_lower = cand_raw.lower()
        cand_norm = _normalize_symbol(cand_raw)
        cand_leaf = cand_raw.split(".")[-1].lower()

        # 1. Exact full or leaf symbol match
        if target_raw == cand_raw or target_lower == cand_lower:
            return 1.0, "exact_symbol"
        if len(target_leaf) >= 4 and target_leaf == cand_leaf:
            best_score = max(best_score, 0.96)
            best_method = "exact_symbol"
            continue

        # 2. Normalized identifier match (e.g., max_retries vs maxRetries / MAX_RETRIES)
        if target_norm and target_norm == cand_norm:
            best_score = max(best_score, 0.94)
            best_method = "fuzzy_symbol"
            continue

        # 3. Substring / prefix / suffix containment for compound identifiers
        if len(target_norm) >= 5 and len(cand_norm) >= 5:
            if target_norm in cand_norm or cand_norm in target_norm:
                overlap = min(len(target_norm), len(cand_norm)) / max(len(target_norm), len(cand_norm))
                score = 0.80 + 0.14 * overlap
                if score > best_score:
                    best_score = score
                    best_method = "fuzzy_symbol"

        # 4. Edit-distance ratio via SequenceMatcher
        ratio = max(
            difflib.SequenceMatcher(None, target_lower, cand_lower).ratio(),
            difflib.SequenceMatcher(None, target_norm, cand_norm).ratio(),
        )
        if ratio >= fuzzy_threshold and ratio > best_score:
            best_score = round(ratio, 4)
            best_method = "fuzzy_symbol"

    return round(best_score, 4), best_method


# ---------------------------------------------------------------------------
# Embedding Backend (`sentence-transformers/all-MiniLM-L6-v2` + Offline Fallback)
# ---------------------------------------------------------------------------

_STEM_PATTERNS = [
    (re.compile(r"ies$"), "y"),
    (re.compile(r"ing$"), ""),
    (re.compile(r"ed$"), ""),
    (re.compile(r"es$"), "e"),
    (re.compile(r"s$"), ""),
]

_CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "timeout": ["seconds", "wait", "hang", "expire", "duration", "latency", "connect", "request"],
    "retry": ["retries", "attempt", "attempts", "backoff", "transient", "fail", "error"],
    "retries": ["retry", "attempt", "attempts", "backoff", "transient", "fail"],
    "rate": ["limit", "throttle", "quota", "requests", "minute", "per"],
    "limit": ["rate", "maximum", "max", "cap", "quota", "throttle"],
    "auth": ["authenticate", "authentication", "token", "bearer", "api", "key", "header", "credentials"],
    "token": ["auth", "authenticate", "bearer", "jwt", "session", "key", "header", "legacy"],
    "batch": ["size", "chunk", "bulk", "items", "payload", "flush"],
    "port": ["host", "listen", "server", "bind", "http", "endpoint"],
    "cache": ["ttl", "expire", "memory", "redis", "store", "evict"],
}


def _lightweight_stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for pat, repl in _STEM_PATTERNS:
        if pat.search(word):
            stemmed = pat.sub(repl, word)
            if len(stemmed) >= 3:
                return stemmed
    return word


class SemanticEmbedder:
    """
    Wraps `sentence-transformers` (`all-MiniLM-L6-v2`) for local dense embeddings.
    Automatically falls back to a deterministic word + character n-gram vector
    representation if `sentence-transformers` is not installed or the model
    weights are unavailable in an offline environment.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._st_model = None
        self.backend_name = "ngram-cosine-fallback"
        self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self) -> None:
        if os.environ.get("DOCDRIFT_DISABLE_ST", "").lower() in ("1", "true", "yes"):
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            try:
                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(self.model_name)
            self.backend_name = f"sentence-transformers/{self.model_name}"
        except Exception:
            self._st_model = None
            self.backend_name = "local-hybrid-minilm-fallback"

    def similarity_matrix(self, doc_texts: Sequence[str], query_texts: Sequence[str]) -> List[List[float]]:
        """
        Computes cosine similarity matrix of shape (len(doc_texts), len(query_texts)).
        """
        if not doc_texts or not query_texts:
            return []

        if self._st_model is not None:
            import numpy as np 

            doc_emb = self._st_model.encode(
                list(doc_texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            qry_emb = self._st_model.encode(
                list(query_texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            sim = np.matmul(doc_emb, qry_emb.T)
            return [[float(max(0.0, min(1.0, val))) for val in row] for row in sim]

        doc_vecs = [self._featurize(t, expand_synonyms=False) for t in doc_texts]
        qry_vecs = [self._featurize(t, expand_synonyms=True) for t in query_texts]
        matrix: List[List[float]] = []
        for dv in doc_vecs:
            row: List[float] = []
            for qv in qry_vecs:
                row.append(self._cosine(dv, qv))
            matrix.append(row)
        return matrix

    @staticmethod
    def _featurize(text: str, expand_synonyms: bool = False) -> Counter:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", " ", text.lower())
        raw_tokens = []
        for tok in cleaned.split():
            raw_tokens.extend(_split_identifier(tok))

        stop_words = {
            "the", "a", "an", "in", "on", "of", "to", "for", "and", "or", "is",
            "are", "was", "were", "be", "by", "with", "from", "as", "at", "it",
            "this", "that", "had", "has", "have", "changed", "function", "setting",
        }
        tokens = [t for t in raw_tokens if t not in stop_words and len(t) > 1]
        feats: Counter = Counter()

        for tok in tokens:
            stem = _lightweight_stem(tok)
            feats[f"w:{stem}"] += 2.0
            feats[f"raw:{tok}"] += 1.5
            if expand_synonyms and stem in _CONCEPT_SYNONYMS:
                for syn in _CONCEPT_SYNONYMS[stem]:
                    feats[f"w:{_lightweight_stem(syn)}"] += 0.85
            if expand_synonyms and tok in _CONCEPT_SYNONYMS:
                for syn in _CONCEPT_SYNONYMS[tok]:
                    feats[f"w:{_lightweight_stem(syn)}"] += 0.85
            padded = f"^{tok}$"
            for i in range(max(0, len(padded) - 3)):
                feats[f"c:{padded[i:i+4]}"] += 0.35

        return feats

    @staticmethod
    def _cosine(vec_a: Counter, vec_b: Counter) -> float:
        if not vec_a or not vec_b:
            return 0.0
        common = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        raw_cos = dot / (norm_a * norm_b)
        return round(min(0.92, math.sqrt(raw_cos) * 0.85), 4)


_DEFAULT_EMBEDDER: Optional[SemanticEmbedder] = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SemanticEmbedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = SemanticEmbedder(model_name=model_name)
    return _DEFAULT_EMBEDDER


# ---------------------------------------------------------------------------
# Main Matching Pipeline
# ---------------------------------------------------------------------------

def match_changes_to_docs(
    symbol_events: Sequence[Dict[str, Any] | SymbolChangeEvent],
    doc_units: Sequence[Dict[str, Any] | DocUnit],
    similarity_threshold: float = 0.38,
    fuzzy_symbol_threshold: float = 0.78,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Matches each code symbol-change event against candidate documentation units.

    Strategy:
      1. First attempts an exact or fuzzy symbol-name match against `mentioned_symbols`.
      2. If no symbol match exceeds `fuzzy_symbol_threshold` for a (doc_unit, event)
         pair, falls back to semantic similarity via sentence embeddings
         (`all-MiniLM-L6-v2`) between `doc_unit["text"]` and a short natural-language
         description of the symbol change.
      3. Returns ranked `(doc_unit, symbol_event, score)` triples above
         `similarity_threshold`, sorted by descending score. Each `doc_unit` dict
         is enriched with `_match_method` ("exact_symbol", "fuzzy_symbol", or
         "semantic_embedding") for downstream inspection.
    """
    events = [SymbolChangeEvent.from_obj(e) for e in symbol_events]
    docs = [DocUnit.from_obj(d) for d in doc_units]

    if not events or not docs:
        return []

    embedder = get_embedder(model_name=model_name)
    doc_texts = [d.text for d in docs]
    change_descriptions = [describe_symbol_event(e) for e in events]

    sem_matrix = embedder.similarity_matrix(doc_texts, change_descriptions)

    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []

    for j, ev in enumerate(events):
        for i, doc in enumerate(docs):
            sym_score, sym_method = compute_symbol_match_score(
                symbol_name=ev.symbol_name,
                mentioned_symbols=doc.mentioned_symbols,
                fuzzy_threshold=fuzzy_symbol_threshold,
            )

            sem_score = sem_matrix[i][j] if sem_matrix else 0.0

            if sym_score >= fuzzy_symbol_threshold:
                final_score = round(min(1.0, sym_score * 0.85 + sem_score * 0.15), 4)
                match_method = sym_method
            else:
                final_score = round(sem_score, 4)
                match_method = "semantic_embedding"

            if final_score >= similarity_threshold:
                doc_dict = doc.to_dict()
                doc_dict["_match_method"] = match_method
                doc_dict["_embedding_backend"] = embedder.backend_name
                ev_dict = ev.to_dict()
                ev_dict["_change_description"] = change_descriptions[j]
                matched_pairs.append((doc_dict, ev_dict, final_score))

    matched_pairs.sort(
        key=lambda item: (
            item[2],
            -int(item[0].get("line_range", [1, 1])[0]),
            item[0].get("file", ""),
        ),
        reverse=True,
    )
    return matched_pairs