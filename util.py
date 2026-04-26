import chromadb

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()  # loads from .env file in current directory


class Embedder:
    def __init__(self, embedding_model_name="text-embedding-3-small", 
                 chunk_size=512, chunk_overlap=64, ):
        print(f"Initializing Embedder with model {embedding_model_name}")
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                separators=["\n\n", "\n", ". ", " "]
                            )
        self.embedding_model_name = embedding_model_name
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=openai_api_key)
        existing_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_names = [ col.name for col in existing_client.list_collections() ]

    def file_transformer(self, file) -> str:
        if file.type in ('text/plain', 'text/markdown'):
            return file.read().decode('utf-8')
        raise ValueError(f"Unsupported type: {file.type}")
        
    def embed_file(self, file) -> tuple[bool, str]:
        print(f"Attempting to embed file: {file.name} of type {file.type}")
        if config.supported_file_types.get(file.type, "unsupported") != "supported":
            print(f"File type {file.type} is not supported.")
            return False, f"File type {file.type} is not supported."

        try:
            print(f"Embedding file: {file.name} of type {file.type}")
            text = self.file_transformer(file)

            chunks_text = self.text_splitter.split_text(text)
            chunks = [
                Document(page_content=t, metadata={"source": file.name})
                for t in chunks_text
            ]
            # sanitized collection for chroma: replace dots and spaces, lowercase
            collection_name = file.name.replace(".", "-").replace(" ", "-").lower()
            _ = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,        # ✅ object not string
                persist_directory="./chroma_db",
                collection_name=collection_name
            )

            self.collection_names.append(collection_name)
            return True, f"File {file.name} embedded successfully."

        except Exception as e:
            return False, f"Embedding failed: {e}"
    