"""
LLM Function Adapters for LightRAG

Provides async LLM functions compatible with LightRAG's requirements
using the existing SteeredLLM infrastructure.
"""

import logging
from typing import Optional, Dict, Any, List
import asyncio

logger = logging.getLogger("machine_poi.llm_adapters")


def create_ollama_adapter(
    model_name: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
    timeout: int = 300,
) -> callable:
    """
    Create an async LLM function for LightRAG using Ollama.

    Args:
        model_name: Ollama model name
        host: Ollama host URL
        timeout: Request timeout

    Returns:
        Async function compatible with LightRAG
    """
    import httpx

    async def ollama_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async Ollama completion for LightRAG."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{host}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": kwargs.get("options", {"num_ctx": 8192}),
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    return ollama_complete


def create_openai_adapter(
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> callable:
    """
    Create an async LLM function for LightRAG using OpenAI API.

    Supports all OpenAI models including:
    - gpt-4o, gpt-4o-mini
    - gpt-5.2, chatgpt-5.2 (latest)
    - o1, o1-mini (reasoning models)

    Args:
        model_name: OpenAI model name (e.g., "gpt-5.2", "gpt-4o-mini")
        api_key: API key (uses OPENAI_API_KEY env var if None)
        base_url: Optional custom base URL for API-compatible services

    Returns:
        Async function compatible with LightRAG
    """
    import os
    from openai import AsyncOpenAI

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def openai_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async OpenAI completion for LightRAG."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )

        return response.choices[0].message.content

    return openai_complete


def create_gemini_adapter(
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> callable:
    """
    Create an async LLM function for LightRAG using Google Gemini API.

    Supports Gemini models including:
    - gemini-2.0-flash, gemini-2.0-flash-lite
    - gemini-3.0-pro, gemini-3.0-ultra (latest)
    - gemini-1.5-pro, gemini-1.5-flash

    Args:
        model_name: Gemini model name (e.g., "gemini-3.0-pro")
        api_key: API key (uses GOOGLE_API_KEY or GEMINI_API_KEY env var if None)

    Returns:
        Async function compatible with LightRAG
    """
    import os
    import google.generativeai as genai

    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    async def gemini_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async Gemini completion for LightRAG."""
        import asyncio

        # Build conversation history
        contents = []

        if history_messages:
            for msg in history_messages:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append({"role": role, "parts": [msg.get("content", "")]})

        contents.append({"role": "user", "parts": [prompt]})

        # Configure model with system instruction
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 2048),
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        # Run in executor since genai is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(contents)
        )

        return response.text

    return gemini_complete


def create_local_llm_adapter(
    steered_llm: "SteeredLLM",
) -> callable:
    """
    Create an async LLM function for LightRAG using existing SteeredLLM.

    This allows using the same local LLM for both steering and entity extraction.

    Args:
        steered_llm: Existing SteeredLLM instance

    Returns:
        Async function compatible with LightRAG
    """

    async def local_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Async wrapper for local LLM."""
        # Build full prompt
        full_prompt = ""

        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"

        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role.capitalize()}: {content}\n"

        full_prompt += f"User: {prompt}\n\nAssistant:"

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: steered_llm.generate(
                prompt=full_prompt,
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
            )
        )

        return result

    return local_complete
