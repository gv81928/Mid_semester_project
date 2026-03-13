from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AgentOutput, UserQuery


class RetrievalAgent(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: UserQuery) -> AgentOutput:
        raise NotImplementedError
