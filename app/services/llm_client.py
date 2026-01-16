"""
LLM Client Abstraction Layer

Provides a unified interface for interacting with different LLM providers:
- OpenAI API
- Ollama (local instance)
- Ollama (remote instance)

This abstraction allows the application to switch between providers
seamlessly based on configuration.
"""

import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import requests
from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response text
        """
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for text.

        Args:
            text: The input text to embed

        Returns:
            numpy array of the embedding vector
        """
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""

    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.chat_model = Config.OPENAI_CHAT_MODEL
        self.embedding_model = Config.OPENAI_EMBEDDING_MODEL
        logger.info(f"Initialized OpenAI client with models: {self.chat_model}, {self.embedding_model}")

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate a chat completion using OpenAI API."""
        try:
            logger.debug(f"Calling OpenAI chat API with {len(messages)} messages")
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding using OpenAI API."""
        try:
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        try:
            logger.info(f"Generating {len(texts)} embeddings via OpenAI")
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding generation failed: {e}")
            raise


class OllamaClient(LLMClient):
    """Ollama client implementation (supports both local and remote)."""

    def __init__(self, base_url: str, api_key: str = None):
        """
        Initialize Ollama client.

        Args:
            base_url: The Ollama API base URL
            api_key: Optional API key for remote instances
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.chat_model = Config.OLLAMA_CHAT_MODEL
        self.embedding_model = Config.OLLAMA_EMBEDDING_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

        logger.info(f"Initialized Ollama client at {self.base_url}")
        logger.info(f"Models: chat={self.chat_model}, embedding={self.embedding_model}")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Ollama requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate a chat completion using Ollama API."""
        try:
            logger.debug(f"Calling Ollama chat API with {len(messages)} messages")

            # Convert messages to Ollama format
            # Ollama expects a single prompt, so we need to format the conversation
            formatted_messages = self._format_messages_for_ollama(messages)

            payload = {
                "model": self.chat_model,
                "messages": formatted_messages,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            assistant_message = result.get('message', {}).get('content', '')

            if not assistant_message:
                raise ValueError("Empty response from Ollama")

            return assistant_message

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat completion failed: {e}")
            raise RuntimeError(f"Ollama API request failed: {e}")
        except Exception as e:
            logger.error(f"Ollama chat completion error: {e}")
            raise

    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for Ollama API.

        Ollama uses the same message format as OpenAI, so we can pass through.
        """
        return messages

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding using Ollama API."""
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }

            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get('embedding')

            if not embedding:
                raise ValueError("Empty embedding from Ollama")

            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embedding generation failed: {e}")
            raise RuntimeError(f"Ollama API request failed: {e}")
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Ollama API."""
        logger.info(f"Generating {len(texts)} embeddings via Ollama")
        embeddings = []

        # Ollama doesn't have batch embedding API, so we process one by one
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.debug(f"Processing embedding {i+1}/{len(texts)}")

            embedding = self.generate_embedding(text)
            embeddings.append(embedding.tolist())

        return embeddings


def create_llm_client() -> LLMClient:
    """
    Factory function to create the appropriate LLM client based on configuration.

    Returns:
        An instance of LLMClient (OpenAI or Ollama)

    Raises:
        ValueError: If the provider is not supported
    """
    provider = Config.LLM_PROVIDER

    logger.info(f"Creating LLM client for provider: {provider}")

    if provider == 'openai':
        return OpenAIClient()
    elif provider == 'ollama_local':
        return OllamaClient(base_url=Config.OLLAMA_LOCAL_URL)
    elif provider == 'ollama_remote':
        return OllamaClient(
            base_url=Config.OLLAMA_REMOTE_URL,
            api_key=Config.OLLAMA_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
