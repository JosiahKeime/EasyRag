import json
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

from .logging_utils import _serialize_messages, logger

load_dotenv()


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
