import os
import re

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

    @staticmethod
    def normalize_chunk_text(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

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

            normalized_chunks = []
            for chunk in chunks_text:
                normalized_chunk = self.normalize_chunk_text(chunk)
                if normalized_chunk:
                    normalized_chunks.append(normalized_chunk)

            chunks = [
                Document(page_content=t, metadata={"source": file.name})
                for t in normalized_chunks
            ]

            collection_name = file.name.replace(".", "-").replace(" ", "-").lower()

            logger.info(
                "Embedding %d normalized chunks for file '%s' into collection '%s'",
                len(chunks),
                file.name,
                collection_name,
            )

            _ = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory="./chroma_db",
                collection_name=collection_name,
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
