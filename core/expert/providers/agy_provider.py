import asyncio
import json
import logging
import os
import shutil
from typing import List, Dict, Optional, Any
from .base import BaseExpertProvider

logger = logging.getLogger(__name__)

class AgyExpertProvider(BaseExpertProvider):
    """
    Expert provider using Google DeepMind Antigravity CLI (agy).
    Supports persistent conversational threads (--conversation <id>),
    project scoping (--project <dir>), and automated tool execution
    (--dangerously-skip-permissions) for computer use, code, and system tasks.
    """

    def __init__(self, agy_path: str = "agy", default_project: Optional[str] = None):
        self.agy_path = agy_path
        self.default_project = default_project or os.environ.get("AGY_DEFAULT_PROJECT", r"C:\Users\lucas\MiniCPM-o-Demo")
        # Maps voice_session_id -> agy_conversation_id
        self._session_map: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "agy"

    def is_available(self) -> bool:
        if os.path.isabs(self.agy_path) and os.path.isfile(self.agy_path):
            return True
        return shutil.which(self.agy_path) is not None

    def get_conversation_id(self, session_id: str) -> Optional[str]:
        return self._session_map.get(session_id)

    def set_conversation_id(self, session_id: str, conversation_id: str) -> None:
        self._session_map[session_id] = conversation_id

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
        session_id: Optional[str] = None,
        project_dir: Optional[str] = None,
    ) -> str:
        # Give AGY at least 60s for full CLI startup, tool loading, and LLM turn
        effective_timeout = max(timeout, 60.0)
        if not self.is_available():
            raise RuntimeError(f"AGY CLI binary not found at '{self.agy_path}'")

        # Check for existing conversation thread for this voice session
        active_conv_id = self._session_map.get(session_id) if session_id else None
        target_project = project_dir or self.default_project

        # Build prompt
        context_parts = []
        default_inst = (
            "Eres el operador experto de Antigravity (AGY) en una llamada de voz. "
            "Tienes acceso completo a la computadora, terminal, código, archivos y subagentes. "
            "Ejecuta la tarea requerida y responde en 1 a 3 oraciones concisas, claras y naturales para ser habladas por voz. "
            "No uses asteriscos, markdown pesado ni bloques de código a menos que sea estrictamente necesario."
        )
        
        # When continuing an existing conversation, we don't need to re-send old history!
        if not active_conv_id:
            context_parts.append(f"[Instrucción del sistema]: {system_prompt or default_inst}")
            if history:
                context_parts.append("[Contexto previo de conversación]:")
                for h in history[-4:]:
                    role = h.get("role", "user")
                    content = h.get("content", "")
                    context_parts.append(f"{role}: {content}")
        else:
            context_parts.append(f"[Instrucción de voz]: Responde en 1 a 3 oraciones breves para TTS.")

        context_parts.append(f"[Tarea a resolver]: {query}")
        full_prompt = "\n".join(context_parts)

        # Build CLI command arguments
        cmd = [
            self.agy_path,
            "-p", full_prompt,
            "--output-format", "json",
            "--dangerously-skip-permissions",
        ]

        if active_conv_id:
            cmd.extend(["--conversation", active_conv_id])
            logger.info(f"[Expert:AGY] Resuming conversation '{active_conv_id}' for session '{session_id}'")
        else:
            logger.info(f"[Expert:AGY] Starting new AGY conversation thread for session '{session_id}'")

        if target_project and os.path.isdir(target_project):
            cmd.extend(["--project", target_project])

        logger.info(f"[Expert:AGY] Executing: {' '.join(cmd[:6])}... (query: '{query[:60]}...')")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError(f"AGY CLI query timed out after {effective_timeout}s")

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()

        # Parse structured JSON output
        parsed_response = ""
        returned_conv_id = None
        if out_text:
            try:
                data = json.loads(out_text)
                parsed_response = data.get("response", "").strip()
                returned_conv_id = data.get("conversation_id")
                status = data.get("status")
                logger.info(
                    f"[Expert:AGY] JSON OK: conv_id={returned_conv_id}, status={status}, "
                    f"tokens={data.get('usage', {}).get('total_tokens')}, cached={data.get('usage', {}).get('cache_read_tokens')}"
                )
            except Exception:
                parsed_response = out_text

        # Record conversation ID for future session turns
        if session_id and returned_conv_id:
            self._session_map[session_id] = returned_conv_id

        if proc.returncode != 0 and not parsed_response:
            raise RuntimeError(f"AGY CLI exited with code {proc.returncode}: {err_text or out_text}")

        final_output = parsed_response or out_text
        logger.info(f"[Expert:AGY] Response received ({len(final_output)} chars): '{final_output[:90]}...'")
        return final_output

