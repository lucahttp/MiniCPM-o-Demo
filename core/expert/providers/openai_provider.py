import asyncio
import logging
from typing import List, Dict, Optional
import httpx
from .base import BaseExpertProvider

logger = logging.getLogger(__name__)

class OpenAIExpertProvider(BaseExpertProvider):
    """Expert provider using OpenAI or any OpenAI-compatible HTTP API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 20.0,
    ) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI API key is not configured (set OPENAI_API_KEY)")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for h in history[-4:]:
                role = "assistant" if h.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": h.get("content", "")})
        messages.append({"role": "user", "content": query})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 512,
        }

        logger.info(f"[Expert:OpenAI] Calling {url} model={self.model}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text}")
            
            data = resp.json()
            choices = data.get("choices")
            if choices and len(choices) > 0:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    logger.info(f"[Expert:OpenAI] Received response ({len(content)} chars)")
                    return content

            raise RuntimeError(f"No valid choices returned: {data}")
