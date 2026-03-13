from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RecommendationTool:
    """Context-aware recommendations based on user profile interests."""

    def __init__(self, profiles_file: str, catalog_file: str) -> None:
        self.profiles_file = Path(profiles_file)
        self.catalog_file = Path(catalog_file)
        self.profiles = self._load_json(self.profiles_file)
        self.catalog = self._load_json(self.catalog_file)

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def recommend(self, user_id: str, top_k: int = 3) -> list[dict[str, Any]]:
        profile = next((p for p in self.profiles if p.get("user_id") == user_id), None)
        if not profile:
            profile = {"user_id": user_id, "interests": ["books", "productivity", "ai"]}

        interests = set(i.lower() for i in profile.get("interests", []))
        scored = []
        for item in self.catalog:
            tags = set(t.lower() for t in item.get("tags", []))
            overlap = len(interests.intersection(tags))
            if overlap > 0:
                scored.append(
                    {
                        "source": item.get("id", "catalog"),
                        "content": f"{item.get('name')}: {item.get('description')}",
                        "score": min(1.0, 0.4 + 0.2 * overlap),
                        "metadata": {"tags": list(tags)},
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
