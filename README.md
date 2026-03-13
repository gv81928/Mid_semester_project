# Multi-Agent Agentic RAG System

This project implements a modular multi-agent Agentic RAG pipeline where a coordinator routes each query to specialized retrieval agents, executes retrieval in parallel, and synthesizes the final response through an Ollama-powered LLM layer (with a deterministic fallback).


## System Architecture

- Coordinator Agent (`RetrievalRouterAgent`):
  - Receives user query.
  - Selects specialized agents using intent signals.
  - Can fan out to all agents if intent is ambiguous.

- Specialized Retrieval Agents:
  - `StructuredRetrievalAgent`:
    - Tool: `TextToSQLTool` (SQLite for demo; can be swapped for PostgreSQL/MySQL adapters).
    - Handles structured analytics and transactional facts.
  - `SemanticRetrievalAgent`:
    - Tools: `InMemoryVectorSearch` over two stores (`Vector Search X` and `Vector Search Y`).
    - Handles unstructured semantic retrieval from documents.
  - `WebRetrievalAgent`:
    - Tool: `WebSearchTool` (DuckDuckGo Instant Answer API).
    - Handles real-time/public information lookup.
  - `RecommendationRetrievalAgent`:
    - Tool: `RecommendationTool` (profile + catalog based).
    - Handles personalized and context-aware suggestions.

- Synthesis Layer (`LLMSynthesizer`):
  - Merges retrieved evidence from all selected agents.
  - Produces final answer + citations + execution debug metadata.
  - Uses local Ollama chat inference (`/api/chat`) by default.
  - Falls back to deterministic synthesis when Ollama is unavailable.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── docs_x.json
│   ├── docs_y.json
│   ├── user_profiles.json
│   └── catalog.json
└── src/
		└── multi_agent_rag/
				├── __init__.py
				├── app.py
				├── llm.py
				├── models.py
				├── orchestrator.py
				├── router.py
				├── agents/
				│   ├── __init__.py
				│   ├── base.py
				│   ├── recommendation_agent.py
				│   ├── semantic_agent.py
				│   ├── structured_agent.py
				│   └── web_agent.py
				└── tools/
						├── recommendation_tool.py
						├── sql_tool.py
						├── vector_search_tool.py
						└── web_search_tool.py
```

## End-to-End Workflow Mapping

### 1) Query Submission

`app.py` receives:

- `--query` (required)
- `--user-id` (optional, default `anonymous`)
- `--data-dir` (optional, default `data`)

### 2) Routing to Specialized Agents

`RetrievalRouterAgent.route()` inspects query intent and selects one or more of:

- `structured`
- `semantic`
- `web`
- `recommendation`

### 3) Parallel Retrieval

`MultiAgentRAGSystem.answer()` executes selected agent retrieval calls concurrently using `ThreadPoolExecutor`.

### 4) Data Integration + LLM Synthesis

`LLMSynthesizer.synthesize()` combines top evidence snippets from each agent and formats a coherent response with source citations.

### 5) Output Generation

CLI outputs JSON:

- `answer`
- `citations`
- `debug` (routing decision, latency context, errors, model fallback info)

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Ollama locally and pull a model:

```bash
ollama serve
ollama pull llama3:latest
```

5. (Optional) Configure Ollama environment variables:

```bash
cp .env.example .env
# set OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS if needed
```

## Run

From the project root:

```bash
PYTHONPATH=src python -m multi_agent_rag.app --query "What is the total revenue and suggest items for alice" --user-id alice
```

Additional examples:

```bash
PYTHONPATH=src python -m multi_agent_rag.app --query "Find latest updates on vector databases" --user-id bob
PYTHONPATH=src python -m multi_agent_rag.app --query "Show customer alice orders from database" --user-id alice
PYTHONPATH=src python -m multi_agent_rag.app --query "Recommend products for carol based on profile" --user-id carol
```

## Ollama Integration

- Default endpoint: `http://localhost:11434/api/chat`
- Default model: `llama3:latest`
- Config vars:
  - `OLLAMA_BASE_URL`
  - `OLLAMA_MODEL`
  - `OLLAMA_TIMEOUT_SECONDS`

If Ollama is down or times out, the system automatically returns a deterministic synthesized response instead of failing.
