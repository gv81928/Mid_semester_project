from __future__ import annotations

import time

from ..models import AgentOutput, RetrievalResult, UserQuery
from ..tools.sql_tool import TextToSQLTool
from .base import RetrievalAgent


class StructuredRetrievalAgent(RetrievalAgent):
    name = "structured"

    def __init__(self, db_path: str) -> None:
        self.tool = TextToSQLTool(db_path)

    def retrieve(self, query: UserQuery) -> AgentOutput:
        start = time.perf_counter()
        try:
            sql = self.tool.text_to_sql(query.text)
            rows = self.tool.run_query(sql)
            results = [
                RetrievalResult(
                    agent=self.name,
                    source="sqlite.sales",
                    content=str(row),
                    score=0.8,
                    metadata={"sql": sql},
                )
                for row in rows[:5]
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
