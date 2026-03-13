from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class CodeSearchTool:
    """Retrieves relevant code snippets from a local folder using token similarity."""

    def __init__(self, code_dir: str, extensions: tuple[str, ...] = (".py", ".js", ".ts", ".md")) -> None:
        self.code_dir = Path(code_dir)
        self.extensions = extensions
        self.documents = self._load_documents()

    def _load_documents(self) -> list[dict[str, Any]]:
        if not self.code_dir.exists():
            return []

        docs: list[dict[str, Any]] = []
        for path in self.code_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            snippet = "\n".join(text.splitlines()[:80])
            docs.append(
                {
                    "source": str(path),
                    "content": snippet,
                    "metadata": {"type": "code", "file_name": path.name},
                }
            )
        return docs

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._tf_vector(_tokenize(query))
        scored: list[dict[str, Any]] = []

        for doc in self.documents:
            d_vec = self._tf_vector(_tokenize(doc["content"]))
            score = self._cosine_similarity(q_vec, d_vec)
            if score > 0:
                scored.append(
                    {
                        "source": doc["source"],
                        "content": doc["content"],
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
