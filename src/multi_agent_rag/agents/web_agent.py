from __future__ import annotations

import time

from ..models import AgentOutput, RetrievalResult, UserQuery
from ..tools.web_search_tool import WebSearchTool
from .base import RetrievalAgent


class WebRetrievalAgent(RetrievalAgent):
    name = "web"

    def __init__(self) -> None:
        self.tool = WebSearchTool()

    def retrieve(self, query: UserQuery) -> AgentOutput:
        start = time.perf_counter()
        try:
            hits = self.tool.search(query.text, top_k=3)
            results = [
                RetrievalResult(
                    agent=self.name,
                    source=h["source"],
                    content=h["content"],
                    score=float(h["score"]),
                    metadata=h.get("metadata", {}),
                )
                for h in hits
            ]
            return AgentOutput(
                agent_name=self.name,
                results=results,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )
        except Exception as exc:
            return AgentOutput(
                agent_name=self.name,
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
