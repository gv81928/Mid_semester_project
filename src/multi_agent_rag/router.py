from __future__ import annotations

from dataclasses import dataclass

from .models import UserQuery


@dataclass
class RoutingDecision:
    selected_agents: list[str]
    rationale: str


class RetrievalRouterAgent:
    """Routes a user query to the most relevant specialized retrieval agents."""

    STRUCTURED_KEYWORDS = {
        "sql",
        "database",
        "table",
        "count",
        "sum",
        "orders",
        "customer",
        "revenue",
    }
    SEMANTIC_KEYWORDS = {"pdf", "document", "manual", "policy", "book", "record", "semantic"}
    WEB_KEYWORDS = {"latest", "today", "news", "current", "web", "online", "real-time", "api"}
    RECO_KEYWORDS = {"recommend", "suggest", "best", "next", "profile", "preference"}

    def route(self, query: UserQuery) -> RoutingDecision:
        text = query.text.lower()
        selected: list[str] = []

        if self._contains_any(text, self.STRUCTURED_KEYWORDS):
            selected.append("structured")
        if self._contains_any(text, self.SEMANTIC_KEYWORDS):
            selected.append("semantic")
        if self._contains_any(text, self.WEB_KEYWORDS):
            selected.append("web")
        if self._contains_any(text, self.RECO_KEYWORDS):
            selected.append("recommendation")

        # Fallback to a broad retrieval fan-out for ambiguous requests.
        if not selected:
            selected = ["structured", "semantic", "web", "recommendation"]
            rationale = "No strong intent signal detected; running all specialized agents."
        else:
            rationale = f"Intent matched agents: {', '.join(selected)}"

        return RoutingDecision(selected_agents=selected, rationale=rationale)

    @staticmethod
    def _contains_any(text: str, keywords: set[str]) -> bool:
        return any(k in text for k in keywords)
