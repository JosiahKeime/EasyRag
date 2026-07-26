import os
import re
from pathlib import Path
from types import SimpleNamespace

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from .logging_utils import logger

load_dotenv()


class Embedder:
    def __init__(self, embedding_model_name=None,
                 chunk_size=512, chunk_overlap=64, chromadb_path="./chroma_db"):
        if embedding_model_name is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

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

    @staticmethod
    def normalize_chunk_text(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _coerce_file_input(file) -> object:
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            ext = path.suffix.lower()
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".markdown": "text/markdown",
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }
            return SimpleNamespace(
                name=path.name,
                type=mime_map.get(ext, "application/octet-stream"),
                read=lambda: path.read_bytes(),
            )
        return file

    def file_transformer(self, file) -> str:
        file_obj = self._coerce_file_input(file)
        if file_obj.type in ("text/plain", "text/markdown"):
            return file_obj.read().decode("utf-8")
        raise ValueError(f"Unsupported type: {file_obj.type}")

    def check_duplicate(self, file) -> bool:
        file_obj = self._coerce_file_input(file)
        collection_name = file_obj.name.replace(".", "-").replace(" ", "-").lower()
        if collection_name in self.collection_names:
            logger.info("Duplicate detected for file '%s' -> collection '%s'", file.name, collection_name)
            return True
        return False

    def check_supported_type(self, file) -> bool:
        file_obj = self._coerce_file_input(file)
        if config.supported_file_types.get(file_obj.type, "unsupported") != "supported":
            logger.warning("Unsupported file type: %s for file %s", file_obj.type, file_obj.name)
            return False
        return True

    def embed_file(self, file) -> tuple[bool, str]:
        file_obj = self._coerce_file_input(file)
        logger.info("Attempting to embed file: %s of type %s", file_obj.name, file_obj.type)

        if self.check_duplicate(file_obj):
            return False, f"Duplicate file: {file_obj.name} already embedded."

        if not self.check_supported_type(file_obj):
            return False, f"File type {file_obj.type} is not supported."

        try:
            text = self.file_transformer(file_obj)
            chunks_text = self.text_splitter.split_text(text)

            normalized_chunks = []
            for chunk in chunks_text:
                normalized_chunk = self.normalize_chunk_text(chunk)
                if normalized_chunk:
                    normalized_chunks.append(normalized_chunk)

            chunks = [
                Document(page_content=t, metadata={"source": file_obj.name})
                for t in normalized_chunks
            ]

            collection_name = file_obj.name.replace(".", "-").replace(" ", "-").lower()

            logger.info(
                "Embedding %d normalized chunks for file '%s' into collection '%s'",
                len(chunks),
                file_obj.name,
                collection_name,
            )

            _ = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory="./chroma_db",
                collection_name=collection_name,
            )

            self.collection_names.append(collection_name)
            return True, f"File {file_obj.name} embedded successfully."

        except Exception as e:
            logger.exception("Embedding failed for file '%s'", file_obj.name)
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
