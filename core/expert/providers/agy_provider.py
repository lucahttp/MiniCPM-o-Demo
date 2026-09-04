import asyncio
import logging
import os
import shutil
from typing import List, Dict, Optional
from .base import BaseExpertProvider

logger = logging.getLogger(__name__)

class AgyExpertProvider(BaseExpertProvider):
    """Expert provider using Google DeepMind Antigravity CLI (agy)."""

    def __init__(self, agy_path: str = "agy"):
        self.agy_path = agy_path

    @property
    def name(self) -> str:
        return "agy"

    def is_available(self) -> bool:
        if os.path.isabs(self.agy_path) and os.path.isfile(self.agy_path):
            return True
        return shutil.which(self.agy_path) is not None

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 25.0,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(f"AGY CLI binary not found at '{self.agy_path}'")

        # Build composite prompt
        context_parts = []
        if system_prompt:
            context_parts.append(f"[Instrucción del sistema]: {system_prompt}")
        if history:
            context_parts.append("[Contexto previo de conversación]:")
            for h in history[-4:]:
                role = h.get("role", "user")
                content = h.get("content", "")
                context_parts.append(f"{role}: {content}")
        context_parts.append(f"[Pregunta del usuario]: {query}")
        context_parts.append("[Respuesta concisa en 1 a 3 oraciones para ser leída por voz]:")

        full_prompt = "\n".join(context_parts)

        logger.info(f"[Expert:AGY] Invoking agy for query: '{query[:80]}...'")
        proc = await asyncio.create_subprocess_exec(
            self.agy_path,
            "-p",
            full_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError(f"AGY CLI query timed out after {timeout}s")

        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 and not output:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"AGY CLI exited with code {proc.returncode}: {err_msg}")

        logger.info(f"[Expert:AGY] Received response ({len(output)} chars): '{output[:100]}...'")
        return output
