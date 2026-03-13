from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserQuery:
    text: str
    user_id: str = "anonymous"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    agent: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    agent_name: str
    results: list[RetrievalResult]
    latency_ms: float
    error: str | None = None


@dataclass
class FinalResponse:
    answer: str
    citations: list[dict[str, Any]]
    debug: dict[str, Any]
