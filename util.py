import logging
import re
from datetime import datetime, timezone
import json
from pathlib import Path
from langchain_core import messages
import chromadb

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from enum import Enum
import os

load_dotenv()  # loads from .env file in current directory


SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "token",
    "authorization",
    "secret",
    "password",
    "access_token",
    "refresh_token",
    "client_secret",
}

API_KEY_PATTERN = re.compile(
    r"(?i)\b("
    r"sk-[A-Za-z0-9]{10,}"
    r"|AIza[0-9A-Za-z\-_]{20,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|Bearer\s+[A-Za-z0-9._\-]+"
    r")\b"
)


def _is_sensitive_key(key):
    if not isinstance(key, str):
        return False
    lowered = key.lower().replace("-", "_")
    return any(name in lowered for name in SENSITIVE_KEY_NAMES)


def redact_sensitive_data(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_key(key) else redact_sensitive_data(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]

    if isinstance(value, str):
        return API_KEY_PATTERN.sub("[REDACTED]", value)

    return value


class RedactingJsonHandler(logging.Handler):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(exist_ok=True)

    def emit(self, record):
        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }

            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)

            with self.file_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(redact_sensitive_data(payload), ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


def configure_logging():
    logger = logging.getLogger("easy_rag")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(exist_ok=True)

    text_log_path = log_dir / "easy_rag.log"
    json_log_path = log_dir / "easy_rag.jsonl"

    file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    json_handler = RedactingJsonHandler(json_log_path)
    json_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(json_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(stream_handler)

    logger.info("Logging initialized | text_log=%s | json_log=%s", text_log_path, json_log_path)
    return logger


logger = configure_logging()


def _serialize_messages(messages_to_log):
    serialized = []
    for message in messages_to_log:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        serialized.append(
            {
                "type": getattr(message, "type", None),
                "content": content,
            }
        )
    return serialized


class Embedder:
    def __init__(self, embedding_model_name="text-embedding-3-small",
                 chunk_size=512, chunk_overlap=64, chromadb_path="./chroma_db"):
        logger.info(
            "Initializing Embedder | model=%s | chunk_size=%s | chunk_overlap=%s | chromadb_path=%s",
            embedding_model_name,
            chunk_size,
            chunk_overlap,
            chromadb_path,
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.embedding_model_name = embedding_model_name
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=openai_api_key)
        self.chromadb_path = chromadb_path
        existing_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection_names = [col.name for col in existing_client.list_collections()]

    def file_transformer(self, file) -> str:
        if file.type in ("text/plain", "text/markdown"):
            return file.read().decode("utf-8")
        raise ValueError(f"Unsupported type: {file.type}")

    def check_duplicate(self, file) -> bool:
        collection_name = file.name.replace(".", "-").replace(" ", "-").lower()
        if collection_name in self.collection_names:
            logger.info("Duplicate detected for file '%s' -> collection '%s'", file.name, collection_name)
            return True
        return False

    def check_supported_type(self, file) -> bool:
        if config.supported_file_types.get(file.type, "unsupported") != "supported":
            logger.warning("Unsupported file type: %s for file %s", file.type, file.name)
            return False
        return True

    def embed_file(self, file) -> tuple[bool, str]:
        logger.info("Attempting to embed file: %s of type %s", file.name, file.type)

        if self.check_duplicate(file):
            return False, f"Duplicate file: {file.name} already embedded."

        if not self.check_supported_type(file):
            return False, f"File type {file.type} is not supported."

        try:
            text = self.file_transformer(file)
            chunks_text = self.text_splitter.split_text(text)
            chunks = [
                Document(page_content=t, metadata={"source": file.name})
                for t in chunks_text
            ]
            collection_name = file.name.replace(".", "-").replace(" ", "-").lower()

            logger.info("Embedding %d chunks for file '%s' into collection '%s'", len(chunks), file.name, collection_name)

            _ = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory="./chroma_db",
                collection_name=collection_name
            )

            self.collection_names.append(collection_name)
            return True, f"File {file.name} embedded successfully."

        except Exception as e:
            logger.exception("Embedding failed for file '%s'", file.name)
            return False, f"Embedding failed: {e}"

    def vector_db_search(self, query, collection_name, k=3):
        logger.info("Vector search | collection=%s | query=%s | k=%s", collection_name, query, k)

        if collection_name not in self.collection_names:
            logger.warning("Collection '%s' does not exist.", collection_name)
            return []

        collection = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.chromadb_path,
        )
        results = collection.similarity_search(query, k=k)
        logger.info("Vector search returned %d results for collection '%s'", len(results), collection_name)
        return results


class Context:
    def __init__(self, history=[], system_prompt=config.system_prompt, documents=[]):
        self.system_prompt = system_prompt
        self.history = history
        self.documents = documents

    def build_documents_context(self, Embedder, query) -> str:
        logger.info("Building document context | query=%s", query)
        doc_context = ""

        for doc in self.documents:
            logger.info("Searching document '%s' for query '%s'", doc, query)
            results = Embedder.vector_db_search(query, doc, k=1)

            if not results:
                logger.info("No results found for document '%s'", doc)
                continue

            doc_context += f"### Document: {doc}\n"
            for i, result in enumerate(results):
                chunk_text = result.page_content
                logger.info("Retrieved chunk %d from document '%s': %s", i + 1, doc, chunk_text)
                doc_context += f"[Chunk {i+1}]\n{chunk_text}\n\n"

        logger.info("Document context built | length=%d | content=%s", len(doc_context), doc_context)
        return doc_context

    def history_to_langchain_messages(self, history) -> list[messages.BaseMessage]:
        msg = []
        for m in history:
            if m["role"] == "user":
                msg.append(HumanMessage(content=m["content"]))
            else:
                msg.append(AIMessage(content=m["content"]))
        return msg

    def build_context(self, history, user_input, doc_context):
        msg = []
        full_system = self.system_prompt + "\n\n## Relevant Document Excerpts\n" + doc_context

        logger.info(
            "Building LLM context | history_entries=%d | user_input=%s | doc_context=%s",
            len(history),
            user_input,
            doc_context,
        )
        logger.info("System prompt + document context: %s", full_system)

        msg.append(SystemMessage(content=full_system))
        msg.extend(self.history_to_langchain_messages(history))
        msg.append(HumanMessage(content=user_input))

        logger.info("Final context messages: %s", json.dumps(_serialize_messages(msg), ensure_ascii=False))
        return msg

    def update_history(self, user_input, response, history):
        history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        history.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})

    def save_history_to_json(self, history: list, path: str = "./converstations/history.json"):
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def load_history_from_json(self, path: str = "./converstations/history.json") -> list:
        file = Path(path)
        if not file.exists():
            return []
        with open(file, "r") as f:
            return json.load(f)


class LLMClient:
    def __init__(self,
                 provider: str = "openai",
                 model: str = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.7,
                 ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-opus-4-5",
        }
        self.model = model or default_models[provider]

        logger.info(
            "Initializing LLMClient | provider=%s | model=%s | max_tokens=%s | temperature=%s",
            self.provider,
            self.model,
            self.max_tokens,
            self.temperature,
        )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=True,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.client = ChatAnthropic(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                streaming=True,
            )

    def invoke(self, messages: list[BaseMessage]) -> str:
        logger.info(
            "LLM invoke start | provider=%s | model=%s | messages=%s",
            self.provider,
            self.model,
            json.dumps(_serialize_messages(messages), ensure_ascii=False),
        )

        try:
            response = self.client.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            logger.info(
                "LLM invoke response | provider=%s | model=%s | response=%s",
                self.provider,
                self.model,
                content,
            )
            return content
        except Exception:
            logger.exception("LLM invoke failed | provider=%s | model=%s", self.provider, self.model)
            raise

    def stream(self, messages: list[BaseMessage]):
        logger.info(
            "LLM stream start | provider=%s | model=%s | messages=%s",
            self.provider,
            self.model,
            json.dumps(_serialize_messages(messages), ensure_ascii=False),
        )

        try:
            for chunk in self.client.stream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    logger.info(
                        "LLM stream chunk | provider=%s | model=%s | chunk=%s",
                        self.provider,
                        self.model,
                        content,
                    )
                    yield content
        except Exception:
            logger.exception("LLM stream failed | provider=%s | model=%s", self.provider, self.model)
            raise

    def get_token_count(self, messages: list[BaseMessage]) -> int:
        return self.client.get_num_tokens_from_messages(messages)
