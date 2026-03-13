from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class InMemoryVectorSearch:
    """A light-weight vector-like search using term-frequency cosine similarity."""

    def __init__(self, data_file: str) -> None:
        self.data_file = Path(data_file)
        self.documents = self._load_documents()

    def _load_documents(self) -> list[dict[str, Any]]:
        if not self.data_file.exists():
            return []
        return json.loads(self.data_file.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._tf_vector(_tokenize(query))
        scored: list[dict[str, Any]] = []

        for doc in self.documents:
            d_vec = self._tf_vector(_tokenize(doc.get("content", "")))
            score = self._cosine_similarity(q_vec, d_vec)
            if score > 0:
                scored.append(
                    {
                        "source": doc.get("source", "unknown"),
                        "content": doc.get("content", ""),
                        "score": score,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tf_vector(tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        n = float(len(tokens))
        return {k: v / n for k, v in counts.items()}

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a).intersection(b)
        dot = sum(a[t] * b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
