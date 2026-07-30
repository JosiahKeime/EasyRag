from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any

from attr import dataclass, field
from langchain_core import messages as langchain_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from util.embedder import Embedder
from .logging_utils import _serialize_messages, logger

@dataclass
class RetrievalContext:
    doc_id: str
    text: str
    score: float
    collection: str
    query: str
    turn_index: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Context:
    history: list[dict[str, Any]] = field(default_factory=list)
    retrieval_contexts: list[RetrievalContext] = field(default_factory=list)

    def append_history(self, role: str, content: Any) -> None:
        self.history.append({"role": role, "content": content})

    def append_retrieval_context(self, doc_id: str, text: str, score: float, collection: str, query: str, turn_index: int) -> None:
        self.retrieval_contexts.append(
            RetrievalContext(
                doc_id=doc_id,
                text=text,
                score=score,
                collection=collection,
                query=query,
                turn_index=turn_index,
            )
        )

    def save_history_to_json(self, path: str = "./converstations/history.json"):
        path_obj = Path(path) + time.now().strftime("_%Y%m%d_%H%M%S.json")
        with open(path_obj, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history_from_json(self, path: str = "./converstations/history.json") -> list:
        file = Path(path)
        if not file.exists():
            return []
        with open(file, "r") as f:
            loaded_history = json.load(f)
            self.history = list(loaded_history)
            return self.history
