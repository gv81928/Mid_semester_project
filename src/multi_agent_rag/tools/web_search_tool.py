from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


class WebSearchTool:
    """Simple web retrieval using DuckDuckGo Instant Answer API."""

    ENDPOINT = "https://api.duckduckgo.com/"

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            results.append(
                {
                    "source": data.get("AbstractURL", "duckduckgo"),
                    "content": abstract,
                    "score": 0.9,
                    "metadata": {"source_type": "abstract"},
                }
            )

        related_topics = data.get("RelatedTopics", [])
        for topic in related_topics:
            if "Text" in topic:
                results.append(
                    {
                        "source": topic.get("FirstURL", "duckduckgo"),
                        "content": topic["Text"],
                        "score": 0.7,
                        "metadata": {"source_type": "related_topic"},
                    }
                )
            elif "Topics" in topic:
                for nested in topic.get("Topics", []):
                    if "Text" in nested:
                        results.append(
                            {
                                "source": nested.get("FirstURL", "duckduckgo"),
                                "content": nested["Text"],
                                "score": 0.65,
                                "metadata": {"source_type": "nested_related_topic"},
                            }
                        )

        return results[:top_k]
