from __future__ import annotations

import argparse
import json

from .orchestrator import MultiAgentRAGSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-agent agentic RAG system.")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--user-id", default="anonymous", help="User ID for personalization")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rag = MultiAgentRAGSystem(data_dir=args.data_dir)
    result = rag.answer(query_text=args.query, user_id=args.user_id)
    print(json.dumps({"answer": result.answer, "citations": result.citations, "debug": result.debug}, indent=2))


if __name__ == "__main__":
    main()
