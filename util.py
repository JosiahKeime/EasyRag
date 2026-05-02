from datetime import datetime
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



class Embedder:
    def __init__(self, embedding_model_name="text-embedding-3-small", 
                 chunk_size=512, chunk_overlap=64, chromadb_path="./chroma_db"):
        print(f"Initializing Embedder with model {embedding_model_name}")
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                separators=["\n\n", "\n", ". ", " "]
                            )
        self.embedding_model_name = embedding_model_name
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=openai_api_key)
        self.chromadb_path = chromadb_path
        existing_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection_names = [ col.name for col in existing_client.list_collections() ]
        

    def file_transformer(self, file) -> str:
        # simple transformer that reads text and markdown files as UTF-8 strings
        if file.type in ('text/plain', 'text/markdown'):
            return file.read().decode('utf-8')
        raise ValueError(f"Unsupported type: {file.type}")
    
    def check_duplicate(self, file) -> bool:
        # check if a collection with the same name (after sanitization) already exists in chroma
        collection_name = file.name.replace(".", "-").replace(" ", "-").lower()
        if collection_name in self.collection_names:
            print(f"Duplicate detected: {file.name} already exists as collection {collection_name}.")
            return True
        return False
    
    def check_supported_type(self, file) -> bool:
        if config.supported_file_types.get(file.type, "unsupported") != "supported":
            print(f"Unsupported file type: {file.type} for file {file.name}")
            return False
        return True
        
    def embed_file(self, file) -> tuple[bool, str]:
        print(f"Attempting to embed file: {file.name} of type {file.type}")
        if self.check_duplicate(file):
            return False, f"Duplicate file: {file.name} already embedded."
        
        if not self.check_supported_type(file):
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
    
    def vector_db_search(self, query, collection_name, k=3):
        if collection_name not in self.collection_names:
            print(f"Collection {collection_name} does not exist.")
            return []
        collection = Chroma(collection_name=collection_name, embedding_function=self.embedding_model, persist_directory=self.chromadb_path)
        results = collection.similarity_search(query, k=k)
        return results

class Context:
    def __init__(self, history = [], system_prompt = config.system_prompt, documents = []):
        self.system_prompt = system_prompt
        self.history = history
        self.documents = documents

    def build_documents_context(self, Embedder, query) -> str:
        doc_context = ""
        for doc in self.documents:
            results = Embedder.vector_db_search(query, doc, k=1)
            
            if not results:
                continue
            doc_context += f"### Document: {doc}\n"
            for i, result in enumerate(results):
                doc_context += f"[Chunk {i+1}]\n{result.page_content}\n\n"

        return doc_context
    
    def history_to_langchain_messages(self, history) -> list[messages.BaseMessage]:
        msg = []
        # history
        for m in history:
            if m["role"] == "user":
                msg.append(HumanMessage(content=m["content"]))
            else:
                msg.append(AIMessage(content=m["content"]))
        return msg
    
    def build_context(self, history, user_input, doc_context):
        msg = []
        # Combine system prompt and doc context into one SystemMessage
        full_system = self.system_prompt + "\n\n## Relevant Document Excerpts\n" + doc_context
        msg.append(SystemMessage(content=full_system))
        # history
        msg.extend(self.history_to_langchain_messages(history))
        # current input
        msg.append(HumanMessage(content=user_input))
        return msg
    
    def update_history(self, user_input, response, history):
        history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        history.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
    
    def save_history_to_json(self, history: list, path: str = "./converstations/history.json"):
        """
        Saves conversation history to a JSON file.
        history is the list of dicts from st.session_state.messages.
        """
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def load_history_from_json(self, path: str = "./converstations/history.json") -> list:
        """
        Loads conversation history from a JSON file.
        Returns empty list if file doesn't exist.
        """
        file = Path(path)
        if not file.exists():
            return []
        with open(file, "r") as f:
            return json.load(f)




class LLMClient:
    def __init__(self,
        provider: str       = 'openai',
        model: str          = None,
        max_tokens: int     = 1024,
        temperature: float  = 0.7,
    ):
        self.provider    = provider
        self.max_tokens  = max_tokens
        self.temperature = temperature

        # ── default models per provider ──
        default_models = {
            'openai':    "gpt-4o",
            'anthropic': "claude-opus-4-5",
        }
        self.model = model or default_models[provider]

        # ── initialise the right client ──
        if provider == 'openai':
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=True,
            )

        elif provider == 'anthropic':
            from langchain_anthropic import ChatAnthropic
            self.client = ChatAnthropic(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                streaming=True,
            )

    def invoke(self, messages: list[BaseMessage]) -> str:
        """
        Sends messages and waits for the full response.
        Returns the response as a plain string.
        """
        response = self.client.invoke(messages)
        return response.content

    def stream(self, messages: list[BaseMessage]):
        """
        Streams the response token by token.
        Use with st.write_stream() in Streamlit.
        """
        for chunk in self.client.stream(messages):
            if chunk.content:
                yield chunk.content

    def get_token_count(self, messages: list[BaseMessage]) -> int:
        """
        Returns the token count for a list of messages
        without actually invoking the model.
        Useful for tracking context window usage.
        """
        return self.client.get_num_tokens_from_messages(messages)
