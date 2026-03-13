from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agents import (
    RecommendationRetrievalAgent,
    SemanticRetrievalAgent,
    StructuredRetrievalAgent,
    WebRetrievalAgent,
)
from .llm import LLMSynthesizer, flatten_errors
from .models import AgentOutput, FinalResponse, UserQuery
from .router import RetrievalRouterAgent


class MultiAgentRAGSystem:
    """Coordinator-driven multi-agent RAG with parallel retrieval and synthesis."""

    def __init__(self, data_dir: str = "data") -> None:
        data_path = Path(data_dir)
        self.router = RetrievalRouterAgent()
        self.agents = {
            "structured": StructuredRetrievalAgent(str(data_path / "structured.db")),
            "semantic": SemanticRetrievalAgent(
                str(data_path / "docs_x.json"),
                str(data_path / "docs_y.json"),
            ),
            "web": WebRetrievalAgent(),
            "recommendation": RecommendationRetrievalAgent(
                str(data_path / "user_profiles.json"),
                str(data_path / "catalog.json"),
            ),
        }
        self.synthesizer = LLMSynthesizer()

    def answer(self, query_text: str, user_id: str = "anonymous") -> FinalResponse:
        query = UserQuery(text=query_text, user_id=user_id)
        decision = self.router.route(query)

        outputs: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=len(decision.selected_agents)) as pool:
            future_map = {
                pool.submit(self.agents[name].retrieve, query): name
                for name in decision.selected_agents
                if name in self.agents
            }
            for future in as_completed(future_map):
                outputs.append(future.result())

        merged = self.synthesizer.synthesize(query, outputs)
        merged.debug.update(
            {
                "routing": decision.selected_agents,
                "routing_rationale": decision.rationale,
                "errors": flatten_errors(outputs),
            }
        )
        return merged
