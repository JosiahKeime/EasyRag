"""
Skills
======

A "skill" is a self-contained capability the agent can call: a name, a
JSON schema describing its inputs (so the model knows what arguments to
pass), and a Python function that does the work.

This file has two parts:
    1. The generic Skill / SkillRegistry abstraction (rarely needs to change)
    2. Concrete skills you register (this is where you'll spend most of
       your time as you add capabilities — ChromaDB search, file I/O,
       API calls, whatever else your agent needs)

Add a new skill by writing one function and one @registry.register(...)
call in `build_default_registry()` below. Nothing in agent.py needs to
change when you do this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .embedder import Embedder


# ---------------------------------------------------------------------------
# 1. The Skill abstraction
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]  # takes **kwargs, returns a string result

    def to_tool_param(self) -> dict[str, Any]:
        """Convert to the format the Anthropic API expects for a tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, **kwargs) -> str:
        try:
            return self.handler(**kwargs)
        except Exception as e:
            # Tool errors should be reported back to the model as text,
            # not raised — that lets Claude see the failure and decide
            # whether to retry, try a different approach, or ask the user.
            return f"Error running skill '{self.name}': {e}"


class SkillRegistry:
    """Holds all skills the agent knows about."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self.embedder = embedder

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable:
        """Decorator form: @registry.register(...) def my_skill(...): ..."""

        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self._skills[name] = Skill(name, description, input_schema, fn)
            return fn

        return decorator

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def as_tool_params(self) -> list[dict[str, Any]]:
        return [s.to_tool_param() for s in self._skills.values()]


# ---------------------------------------------------------------------------
# 2. Concrete skills
# ---------------------------------------------------------------------------

def build_default_registry(embedder: Embedder | None = None) -> SkillRegistry:
    registry = SkillRegistry(embedder=embedder)

    # --- ChromaDB search skill ------------------------------------------
    # Swap in your real easy_rag collection here. Kept lazy (imported
    # inside the function) so this file still works even if chromadb
    # isn't installed / configured for a quick test.
    @registry.register(
        name="search_knowledge_base",
        description=(
            "Search the local ChromaDB vector database for passages "
            "relevant to a query. Use this before answering questions "
            "that might be covered by the user's own documents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {
                    "type": "integer",
                    "description": "How many chunks to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )
    def search_knowledge_base(query: str, n_results: int = 5) -> str:
        import chromadb

        active_embedder = registry.embedder
        if active_embedder is None:
            active_embedder = Embedder()

        try:
            client = chromadb.PersistentClient(path=active_embedder.chromadb_path)
            active_embedder.collection_names = [collection.name for collection in client.list_collections()]
        except Exception as exc:
            return f"Error refreshing knowledge base collections: {exc}"

        if not active_embedder.collection_names:
            return "No embedded collections found."

        collected_docs: list[str] = []
        for collection_name in active_embedder.collection_names:
            try:
                results = active_embedder.vector_db_search(query, collection_name, k=n_results)
            except Exception as exc:
                collected_docs.append(f"[{collection_name}] Error: {exc}")
                continue

            if not results:
                continue

            for result in results:
                chunk_text = getattr(result, "page_content", "")
                if chunk_text:
                    collected_docs.append(f"[{collection_name}] {chunk_text}")

        if not collected_docs:
            return "No results found."

        return "\n\n---\n\n".join(collected_docs[:n_results * 2])

    # --- A trivial example "skill" to show the pattern for anything else -
    @registry.register(
        name="get_word_count",
        description="Count words in a piece of text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def get_word_count(text: str) -> str:
        return str(len(text.split()))

    return registry