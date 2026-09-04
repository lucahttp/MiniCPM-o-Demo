import asyncio
import logging
import os
import shutil
from typing import List, Dict, Optional
from .base import BaseExpertProvider

logger = logging.getLogger(__name__)

class ClaudeExpertProvider(BaseExpertProvider):
    """Expert provider using Claude Code CLI (claude)."""

    def __init__(self, claude_path: str = "claude"):
        self.claude_path = claude_path

    @property
    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        if os.path.isabs(self.claude_path) and os.path.isfile(self.claude_path):
            return True
        return shutil.which(self.claude_path) is not None

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 25.0,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(f"Claude CLI binary not found at '{self.claude_path}'")

        context_parts = []
        if system_prompt:
            context_parts.append(f"[System Instruction]: {system_prompt}")
        if history:
            context_parts.append("[Previous Conversation]:")
            for h in history[-4:]:
                role = h.get("role", "user")
                content = h.get("content", "")
                context_parts.append(f"{role}: {content}")
        context_parts.append(f"[User Query]: {query}")
        context_parts.append("[Spoken voice response]:")

        full_prompt = "\n".join(context_parts)

        logger.info(f"[Expert:Claude] Invoking claude for query: '{query[:80]}...'")
        # On Windows, claude -p needs stdin disconnected (DEVNULL)
        proc = await asyncio.create_subprocess_exec(
            self.claude_path,
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
            raise TimeoutError(f"Claude CLI query timed out after {timeout}s")

        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 and not output:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Claude CLI exited with code {proc.returncode}: {err_msg}")

        logger.info(f"[Expert:Claude] Received response ({len(output)} chars): '{output[:100]}...'")
        return output
