from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core import messages as langchain_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from .logging_utils import _serialize_messages, logger


class Context:
    def __init__(self, history=None, documents=None):
        self.history = list(history or [])
        self.documents = documents or []

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.history

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self.history = list(value or [])

    def get_messages(self) -> list[dict[str, Any]]:
        return self.history

    def append_message(self, role: str, content: Any) -> None:
        self.history.append({"role": role, "content": content})

    @staticmethod
    def _to_collection_name(document_name):
        if not document_name:
            return None
        name = str(document_name).strip()
        return name.replace(".", "-").replace(" ", "-").lower()

    def build_documents_context(self, embedder, query) -> str:
        logger.info("Building document context | query=%s", query)
        doc_context = ""

        candidate_documents = self.documents or embedder.collection_names
        if not candidate_documents:
            logger.info("No document collections available to search")
            return ""

        for doc in candidate_documents:
            collection_name = self._to_collection_name(doc)
            logger.info(
                "Searching document '%s' (collection '%s') for query '%s'",
                doc,
                collection_name,
                query,
            )

            if not collection_name:
                logger.warning("Skipping empty document name")
                continue

            if collection_name not in embedder.collection_names:
                logger.warning(
                    "Collection '%s' does not exist for document '%s'",
                    collection_name,
                    doc,
                )
                continue

            results = embedder.vector_db_search(query, collection_name, k=1)

            if not results:
                logger.info("No results found for collection '%s'", collection_name)
                continue

            doc_context += f"### Document: {doc}\n"
            for i, result in enumerate(results):
                chunk_text = result.page_content
                logger.info(
                    "Retrieved chunk %d from document '%s': %s",
                    i + 1,
                    doc,
                    chunk_text,
                )
                doc_context += f"[Chunk {i+1}]\n{chunk_text}\n\n"

        logger.info("Document context built | length=%d | content=%s", len(doc_context), doc_context)
        return doc_context

    def history_to_langchain_messages(self, history) -> list[langchain_messages.BaseMessage]:
        msg = []
        for m in history:
            if m["role"] == "user":
                msg.append(HumanMessage(content=m["content"]))
            else:
                msg.append(AIMessage(content=m["content"]))
        return msg

    def build_full_context(self, history=None, user_input="", doc_context=""):
        msg = []
        history = list(history if history is not None else self.history)
        doc_context = "\n\n## Relevant Document Excerpts\n" + doc_context

        logger.info(
            "Building LLM context | history_entries=%d | user_input=%s | doc_context=%s",
            len(history),
            user_input,
            doc_context,
        )
        logger.info("document context: %s", doc_context)

        msg.append(SystemMessage(content=doc_context))
        msg.extend(self.history_to_langchain_messages(history))
        msg.append(HumanMessage(content=user_input))

        logger.info("Final context messages: %s", json.dumps(_serialize_messages(msg), ensure_ascii=False))
        return msg

    def update_history(self, user_input, response, history=None):
        target_history = history if history is not None else self.history
        target_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        target_history.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
        return target_history

    def save_history_to_json(self, history: list | None = None, path: str = "./converstations/history.json"):
        target_history = history if history is not None else self.history
        with open(path, "w") as f:
            json.dump(target_history, f, indent=2)

    def load_history_from_json(self, path: str = "./converstations/history.json") -> list:
        file = Path(path)
        if not file.exists():
            return []
        with open(file, "r") as f:
            loaded_history = json.load(f)
            self.history = list(loaded_history)
            return self.history
