from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .models import AgentOutput, FinalResponse, UserQuery


class LLMSynthesizer:
    """Synthesizes retrieved evidence into a concise response.

    This implementation prefers local Ollama inference and falls back to a deterministic
    synthesis strategy if Ollama is unavailable.
    """

    def __init__(self) -> None:
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3:latest")
        self.timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "25"))

    def synthesize(self, query: UserQuery, outputs: list[AgentOutput]) -> FinalResponse:
        snippets, citations = self._collect_evidence(outputs)
        llm_answer = self._ollama_synthesis(query, snippets)

        if llm_answer is not None:
            return FinalResponse(
                answer=llm_answer,
                citations=citations,
                debug={
                    "llm_used": True,
                    "model": self.model,
                    "provider": "ollama",
                    "agent_count": len(outputs),
                    "evidence_items": len(snippets),
                },
            )

        fallback = self._deterministic_synthesis(query, snippets, citations)
        fallback.debug.update(
            {
                "llm_used": False,
                "model": "deterministic-fallback",
                "provider": "none",
            }
        )
        return fallback

    def _collect_evidence(
        self,
        outputs: list[AgentOutput],
    ) -> tuple[list[str], list[dict[str, str | float]]]:
        snippets: list[str] = []
        citations: list[dict[str, str | float]] = []

        for output in outputs:
            top_results = sorted(output.results, key=lambda r: r.score, reverse=True)[:2]
            for result in top_results:
                snippets.append(f"[{output.agent_name}] {result.content}")
                citations.append(
                    {
                        "agent": output.agent_name,
                        "source": result.source,
                        "score": round(result.score, 4),
                    }
                )

        return snippets, citations

    def _ollama_synthesis(self, query: UserQuery, snippets: list[str]) -> str | None:
        if not snippets:
            return None

        prompt = self._build_prompt(query.text, snippets)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a synthesis assistant for a multi-agent RAG system. "
                        "Produce a concise, factual answer grounded only in provided evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

        message = parsed.get("message", {})
        content = message.get("content", "")
        content = content.strip()
        return content or None

    def _build_prompt(self, query_text: str, snippets: list[str]) -> str:
        evidence = "\n".join(f"- {s}" for s in snippets[:10])
        return (
            f"User Query:\n{query_text}\n\n"
            "Evidence from retrieval agents:\n"
            f"{evidence}\n\n"
            "Instructions:\n"
            "1) Answer the query using only the evidence.\n"
            "2) If evidence is insufficient, explicitly say what is missing.\n"
            "3) Keep the answer concise and actionable."
        )

    def _deterministic_synthesis(
        self,
        query: UserQuery,
        snippets: list[str],
        citations: list[dict[str, str | float]],
    ) -> FinalResponse:

        if snippets:
            evidence_block = "\n".join(f"- {line}" for line in snippets[:8])
            answer = (
                f"Query: {query.text}\n\n"
                "Integrated Answer:\n"
                "I combined evidence from specialized retrieval agents. "
                "Key findings are:\n"
                f"{evidence_block}"
            )
        else:
            answer = (
                f"Query: {query.text}\n\n"
                "No evidence was retrieved. Please refine the question or provide additional context."
            )

        return FinalResponse(
            answer=answer,
            citations=citations,
            debug={
                "evidence_items": len(snippets),
            },
        )


def flatten_errors(outputs: Iterable[AgentOutput]) -> list[str]:
    return [f"{o.agent_name}: {o.error}" for o in outputs if o.error]
