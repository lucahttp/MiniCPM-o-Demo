import asyncio
import json
import logging
from typing import List, Dict, Optional, Any, Callable, Awaitable
import httpx
from .base import BaseExpertProvider

logger = logging.getLogger(__name__)

# Standard function calling tools exposed to the Slow Loop Brain
SLOW_LOOP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_agy",
            "description": (
                "Delega tareas que involucren operar la computadora, ejecutar comandos de terminal (PowerShell/Bash), "
                "inspeccionar o editar código, correr tests, o manejar proyectos y archivos al agente autónomo Antigravity (AGY)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "La instrucción o tarea técnica concreta que debe ejecutar AGY."
                    },
                    "project_dir": {
                        "type": "string",
                        "description": "Directorio del proyecto si aplica (ej. C:\\Users\\lucas\\MiniCPM-o-Demo)."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evalúa expresiones matemáticas exactas (aritmética, potencias, raíces cuadradas, porcentajes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Expresión matemática a evaluar (ej: 'sqrt(24)', '450 * 18')."
                    }
                },
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "time_date",
            "description": "Devuelve la fecha y hora actual exacta del sistema o ubicación especificada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Ubicación o zona horaria opcional (ej: 'Buenos Aires', 'Tokyo')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem",
            "description": "Operaciones con el sistema de archivos: read, write, list, info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list", "info"],
                        "description": "Acción a realizar sobre el sistema de archivos."
                    },
                    "path": {
                        "type": "string",
                        "description": "Ruta de archivo o carpeta."
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenido para la acción 'write'."
                    }
                },
                "required": ["action", "path"]
            }
        }
    }
]


class OpenAIExpertProvider(BaseExpertProvider):
    """
    Expert provider using OpenAI or any OpenAI-compatible HTTP API
    (Groq, MiniMax, OpenAI, DeepSeek, Ollama, etc.).
    Supports native Tool Calling (function calling) to orchestrate
    instant tools and deep AGY delegation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        api_base: str = "https://api.groq.com/openai/v1",
        tool_executor: Optional[Callable[[str, Dict[str, Any], Optional[str]], Awaitable[str]]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.tool_executor = tool_executor

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def set_tool_executor(self, executor: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[str]]) -> None:
        self.tool_executor = executor

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
        session_id: Optional[str] = None,
    ) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI-compatible API key is not configured")

        default_sys = (
            "Eres el director cognitivo de un asistente de voz en tiempo real. "
            "Responde de forma clara, directa y natural en 1 o 2 oraciones para ser leída por voz. "
            "Si la consulta requiere operar la computadora, ejecutar comandos de terminal, inspeccionar o editar código, "
            "llama a la herramienta 'delegate_to_agy'. Si requiere matemáticas exactas, llama a 'calculator'. "
            "Para preguntas de conocimiento, explicaciones o diálogo general, responde DIRECTAMENTE sin herramientas."
        )

        messages = [{"role": "system", "content": system_prompt or default_sys}]
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
            "temperature": 0.3,
            "max_tokens": 512,
            "tools": SLOW_LOOP_TOOLS if self.tool_executor else None,
        }
        if not self.tool_executor:
            payload.pop("tools", None)

        logger.info(f"[Expert:SlowLoop] Calling {url} model={self.model} (tools={'yes' if self.tool_executor else 'no'})")
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Slow Loop API returned HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            choices = data.get("choices")
            if not choices or len(choices) == 0:
                raise RuntimeError(f"No choices in API response: {data}")

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls")

            # 1. Si el modelo no llamó a ninguna herramienta, devolver la respuesta directa (< 300 ms)
            if not tool_calls:
                content = message.get("content", "").strip()
                logger.info(f"[Expert:SlowLoop] Direct response ({len(content)} chars): '{content[:70]}...'")
                return content

            # 2. Si el modelo llamó a herramientas, ejecutarlas
            logger.info(f"[Expert:SlowLoop] Model requested {len(tool_calls)} tool calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
            messages.append(message)

            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name")
                call_id = tc.get("id", "call_1")
                try:
                    fn_args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    fn_args = {}

                logger.info(f"[Expert:SlowLoop] Executing tool '{fn_name}' with args {fn_args}")
                tool_output = ""
                if self.tool_executor:
                    try:
                        tool_output = await self.tool_executor(fn_name, fn_args, session_id)
                    except Exception as e:
                        logger.error(f"[Expert:SlowLoop] Error executing tool {fn_name}: {e}", exc_info=True)
                        tool_output = f"Error ejecutando {fn_name}: {str(e)}"
                else:
                    tool_output = f"Herramienta {fn_name} no disponible."

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": fn_name,
                    "content": str(tool_output),
                })

            # 3. Llamada de seguimiento para que el modelo elabore la respuesta conversacional final
            followup_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 256,
            }
            logger.info(f"[Expert:SlowLoop] Sending follow-up for voice synthesis after tool execution...")
            followup_resp = await client.post(url, headers=headers, json=followup_payload)
            if followup_resp.status_code == 200:
                followup_data = followup_resp.json()
                f_choices = followup_data.get("choices")
                if f_choices and len(f_choices) > 0:
                    final_speech = f_choices[0].get("message", {}).get("content", "").strip()
                    logger.info(f"[Expert:SlowLoop] Synthesized voice response: '{final_speech[:90]}...'")
                    return final_speech

            # Fallback al resultado crudo de la tool si la síntesis falla
            return str(tool_output)

