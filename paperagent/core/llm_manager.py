"""
LLM Manager - Unified interface for multiple LLM providers
"""

from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
import requests

from paperagent.core.config import settings
from loguru import logger


class BaseLLM(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        openai.api_key = self.api_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, max_tokens, temperature, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion"""
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.temperature,
                top_p=kwargs.get("top_p", settings.top_p),
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM provider"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion"""
        try:
            # Extract system message if present
            system_prompt = ""
            chat_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    chat_messages.append(msg)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                temperature=temperature or settings.temperature,
                system=system_prompt,
                messages=chat_messages,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider"""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        try:
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or settings.temperature,
                    "num_predict": max_tokens or settings.max_tokens,
                    "top_p": kwargs.get("top_p", settings.top_p),
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion"""
        try:
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or settings.temperature,
                    "num_predict": max_tokens or settings.max_tokens,
                    "top_p": kwargs.get("top_p", settings.top_p),
                }
            }

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class LLMManager:
    """
    Unified LLM manager supporting multiple providers

    Usage:
        llm = LLMManager()
        response = llm.generate("Write a research paper abstract about AI")
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.default_llm_provider
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> BaseLLM:
        """Initialize LLM based on provider"""
        if self.provider == "openai":
            return OpenAILLM()
        elif self.provider == "anthropic":
            return AnthropicLLM()
        elif self.provider == "ollama":
            return OllamaLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        return self.llm.generate(prompt, system_prompt, max_tokens, temperature, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Chat completion"""
        return self.llm.chat(messages, max_tokens, temperature, **kwargs)

    def switch_provider(self, provider: str):
        """Switch LLM provider"""
        self.provider = provider
        self.llm = self._initialize_llm()
        logger.info(f"Switched to {provider} provider")


# Global LLM manager instance
llm_manager = LLMManager()
