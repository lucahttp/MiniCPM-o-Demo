import asyncio
import logging
import random
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from .config import ExpertConfig, ExpertProvider
from .providers.base import BaseExpertProvider
from .providers.agy_provider import AgyExpertProvider
from .providers.claude_provider import ClaudeExpertProvider
from .providers.minimax_provider import MiniMaxExpertProvider
from .providers.openai_provider import OpenAIExpertProvider

logger = logging.getLogger(__name__)

# Trigger keywords for delegation (Spanish and English explicit intent)
_EXPLICIT_TRIGGERS = [
    # Spanish explicit delegation intent (both user requesting and assistant announcing)
    r"\b(d[eé]jame|voy\s+a|un\s+segundo,?\s+voy\s+a|espera\s+que)\s+(consultar|preguntar|averiguar|pedirle|buscar)\s+(con\s+el\s+|al?\s+)?(experto|supervisor|agy|claude|minimax|groq)\b",
    r"\b(consult[aá](ndo|r[eé])?|pregunt[aá](ndo|r[eé])?)\s+(al?\s+|con\s+el\s+)?(experto|supervisor|agy|claude|minimax|groq)\b",
    r"\b(preg[uú]ntale?\s+a|consult[aá](le)?\s+al?)\s+(agy|claude|minimax|groq|experto)\b",
    r"\b(le\s+pregunto\s+al?\s+(experto|supervisor|agy|claude|minimax|groq))\b",
    r"\b(consultando\s+al?\s+(experto|supervisor|agy|claude|minimax|groq))\b",
    r"\b(llam[aá](r|le)?|pas[aá](r|le)?|us[aá](r)?|deleg[aá](r)?)\s+(al?\s+|con\s+el\s+)?(experto|supervisor|agy|claude|minimax|groq)\b",
    r"\b(experto|supervisor)\s+(para\s+que|que\s+nos\s+ayude|que\s+lo\s+haga)\b",

    # English explicit delegation intent (both user requesting and assistant announcing)
    r"\b(let\s+me|i('ll|\s+will)|one\s+second,?\s+i'll)\s+(check|ask|consult|find\s+out|look\s*up)\s+(with\s+the\s+|the\s+)?(expert|supervisor|agy|claude|minimax|groq)\b",
    r"\b(consulting|asking)\s+(with\s+)?(the\s+)?(expert|supervisor|agy|claude|minimax|groq)\b",
    r"\b(i'm\s+asking|checking\s+with)\s+(the\s+)?(expert|supervisor|agy|claude|minimax|groq)\b",
    r"\b(can\s+you\s+|could\s+you\s+|please\s+)?(call|ask|consult|delegate|delayed|pass\s+(it\s+)?to|use)\s+(to\s+)?(the\s+)?(expert|supervisor|agy|claude|minimax|groq)\b",
    r"\b(call\s+the\s+expert|call\s+the\s+supervisor|delegate\s+to\s+the\s+expert|delayed\s+to\s+the\s+expert)\b",
    r"\b(ask(ing)?|consult(ing)?)\s+(with\s+)?(the\s+)?(expert|supervisor|agy|claude|minimax|groq)\b",

    # Internet / web search / places & business lookups / specifications
    r"\b(search|look\s*up|find)\s+(across|on|in)?\s*(the\s+)?(internet|web|online)\b",
    r"\b(can\s+you\s+|could\s+you\s+|please\s+)?(search|look\s*up)\b",
    r"\b(search\s+it|search\s+for\s+it|look\s+it\s+up|search\s+some\s+place|search\s+a\s+place)\b",
    r"\b(where\s+(can\s+i|to)\s+(find|get|buy|repair|fix|take))\b",
    r"\b(busc[aá](r|me|lo|la)?|averigu[aá](r|me|lo|la)?)\b",
    r"\b(d[oó]nde\s+(puedo|queda|hay|conseguir|arreglar|reparar|comprar))\b",
    r"\b(let\s+me|i('ll|\s+will)|i\s+can)\s+(search|look\s*up|find)\b",
    r"\b(d[eé]jame|voy\s+a|puedo)\s+(buscar|averiguar)\b",
    r"\b(specifications|specs|especificaciones|manual|datasheet)\s+(of|for|de|para)?\b",

    # Complex programming / creation requests that need frontier expert
    r"\b(develop|build|create|code|program)\s+(a\s+|an\s+)?(website|web\s+app|application|script|program|backend|frontend)\b",
    
    # Calculations, taxes, and salary queries (English & Spanish)
    r"\b(calculate|computing|estimate|calcul[aá]|estim[aá])\s+(my\s+|the\s+|our\s+)?(salary|tax|taxes|income|net|gross|sueldo|salario|impuesto|impuestos|ganancias)\b",
    r"\b(help\s+me\s+(to\s+)?(calculate|compute|estimate|figure\s+out)|ay[uú]dame\s+a\s+calcular)\b",
    r"\b(tax\s+system|tax\s+rates?|taxes\s+in|income\s+tax|tax\s+bracket)\b",
    r"\b(calculate\s+(that|this|it)|how\s+much\s+is\s+that|calcul[aá]\s+(eso|esto))\b",
    r"\b(impuestos?|ganancias|salario\s+neto|net\s+salary|sueldo\s+neto|monotributo|jubilaci[oó]n|deducciones|retenciones)\b",
]

_DELEGATE_TAG_REGEX = re.compile(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", re.IGNORECASE)

from .tools import ToolDispatcher
from .parser import fc2dict, MiniCPMParser, resolve_ast_call
from .audio_gate import AudioStreamGate, GateState, ToolCallEvent
from .mcp_client import MCPClient, ToolCallResult

class ExpertSupervisor:
    """Orchestrates hybrid voice delegation to specialized frontier agents (AGY, Claude, MiniMax) and MCP tools."""

    def __init__(self, config: Optional[ExpertConfig] = None, mcp_client: Optional[MCPClient] = None):
        self.config = config or ExpertConfig()
        self.tool_dispatcher = ToolDispatcher()
        self.mcp_client: Optional[MCPClient] = mcp_client

        # Provider initialization
        agy_provider = AgyExpertProvider(
            agy_path=self.config.agy_path,
            default_project=self.config.agy_default_project,
        )
        minimax_provider = MiniMaxExpertProvider(
            api_key=self.config.minimax_api_key,
            group_id=self.config.minimax_group_id,
            model=self.config.minimax_model,
            api_base=self.config.minimax_api_base,
        )
        openai_provider = OpenAIExpertProvider(
            api_key=self.config.openai_api_key or self.config.slow_loop_api_key,
            model=self.config.openai_model,
            api_base=self.config.openai_api_base or "https://api.openai.com/v1",
        )
        groq_provider = OpenAIExpertProvider(
            api_key=self.config.groq_api_key or self.config.slow_loop_api_key,
            model=self.config.groq_model,
            api_base=self.config.groq_api_base,
        )

        # Wire Slow Loop Brain tool executor to AGY and local tools
        async def _slow_loop_tool_executor(fn_name: str, args: Dict[str, Any], session_id: Optional[str] = None) -> str:
            if fn_name == "delegate_to_agy":
                task = args.get("task", "") or str(args)
                project = args.get("project_dir") or self.config.agy_default_project
                logger.info(f"[SlowLoopBrain] Delegating to AGY: '{task[:80]}' (session={session_id}, project={project})")
                return await agy_provider.execute(
                    query=task,
                    timeout=self.config.timeout_seconds,
                    session_id=session_id,
                    project_dir=project,
                )
            # Local instant tools: calculator, time_date, filesystem
            logger.info(f"[SlowLoopBrain] Executing local tool '{fn_name}'")
            return await self.execute_tool_async(fn_name, args)

        groq_provider.set_tool_executor(_slow_loop_tool_executor)
        openai_provider.set_tool_executor(_slow_loop_tool_executor)

        self.providers: Dict[str, BaseExpertProvider] = {
            ExpertProvider.AGY.value: agy_provider,
            ExpertProvider.CLAUDE.value: ClaudeExpertProvider(self.claude_path_clean(self.config.claude_path)),
            ExpertProvider.MINIMAX.value: minimax_provider,
            ExpertProvider.OPENAI.value: openai_provider,
            ExpertProvider.GROQ.value: groq_provider,
        }

    def set_mcp_client(self, client: MCPClient) -> None:
        """Attach an active MCPClient to the supervisor."""
        self.mcp_client = client

    def create_audio_gate(self, sample_rate: int = 16000, mute_as_silence: bool = False) -> AudioStreamGate:
        """Creates a new instance of AudioStreamGate."""
        return AudioStreamGate(sample_rate=sample_rate, mute_as_silence=mute_as_silence)

    @staticmethod
    def claude_path_clean(path: str) -> str:
        return path

    def is_enabled(self) -> bool:
        return self.config.enabled

    def get_provider(self, name: Optional[str] = None) -> BaseExpertProvider:
        prov_key = (name or self.config.provider.value).lower()
        provider = self.providers.get(prov_key)
        if not provider:
            # Fallback to agy or first available
            for p in self.providers.values():
                if p.is_available():
                    return p
            return self.providers[ExpertProvider.AGY.value]
        return provider

    def detect_delegation_intent(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if text contains delegation tags or complex inquiry.
        Returns (should_delegate, extracted_query, optional_provider_override)
        """
        if not self.config.enabled or not text:
            return False, None, None

        # 1. Check for structured tag: [EXPERT: query] or [DELEGATE: query]
        tag_match = _DELEGATE_TAG_REGEX.search(text)
        if tag_match:
            query = tag_match.group(1).strip()
            return True, query, None

        # 2. Check for explicit trigger phrases
        clean = text.lower().strip()
        provider_override = None
        if "agy" in clean:
            provider_override = "agy"
        elif "claude" in clean:
            provider_override = "claude"
        elif "minimax" in clean:
            provider_override = "minimax"
        elif "groq" in clean:
            provider_override = "groq"

        for pattern in _EXPLICIT_TRIGGERS:
            if re.search(pattern, clean):
                q = self.clean_query_text(text)
                return True, q, provider_override

        # 3. If auto_delegate is enabled and query is substantive (> 30 chars and contains question marks or inquiry)
        if self.config.auto_delegate:
            is_question = "?" in text or "¿" in text or clean.startswith(("qué", "que", "cómo", "como", "por qué", "porque", "cuál", "cual", "quién", "quien", "how", "what", "why", "where", "who"))
            is_longer = len(clean.split()) >= 4
            is_not_small_talk = not any(w in clean for w in ["hola", "buen dia", "buenas", "chau", "adios", "gracias", "ok", "dale", "si", "no"])
            if is_question and is_longer and is_not_small_talk:
                q = self.clean_query_text(text)
                return True, q, provider_override

        return False, None, None

    def detect_tool_or_expert(self, text: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detecta si hay una herramienta rápida, una llamada MCP o un experto.
        Soporta parser nativo (<|tool_call_start|>), XML (<function=...>) y legacy ([TOOL:...]).
        Retorna (tipo, nombre_herramienta_o_query, argumentos).
        Tipo puede ser "TOOL", "MCP", "EXPERT" o "NONE".
        """
        if not text:
            return "NONE", None, None

        # 1. Verificar si hay Native OpenBMB o XML tool call vía parser fc2dict
        try:
            _, native_calls = fc2dict(text)
            if native_calls:
                call = native_calls[0]
                t_name = call.get("name", "tool")
                t_args = call.get("arguments", {})
                if t_name.lower() in ("expert", "delegate", "supervisor"):
                    q = t_args.get("query") or t_args.get("__positional__") or text
                    return "EXPERT", q, t_args
                if hasattr(self.tool_dispatcher.registry, t_name):
                    return "TOOL", t_name, t_args
                if self.mcp_client:
                    return "MCP", t_name, t_args
                return "TOOL", t_name, t_args
        except Exception as e:
            logger.debug(f"[ExpertSupervisor] fc2dict parse check failed: {e}")

        # 2. Fallback a tool_dispatcher para formato bracket [TOOL: ...] y [EXPERT: ...]
        return self.tool_dispatcher.detect_tool_or_expert(text)

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Ejecuta y formatea la respuesta en 1 oración limpia lista para que MiniCPM-o la hable de inmediato.
        """
        return self.tool_dispatcher.execute_tool(tool_name, args)

    async def execute_tool_async(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Ejecuta una herramienta ya sea instantánea (<50ms), vía MCP o delegada al experto.
        Formatea el resultado para voz natural.
        """
        if hasattr(self.tool_dispatcher.registry, tool_name):
            ans = await self.tool_dispatcher.execute_tool_async(tool_name, args)
            if ans.startswith("[EXPERT:"):
                q = ans[8:-1].strip()
                res = await self.execute(q)
                return res.get("text", "No pude obtener la respuesta del experto.")
            return ans

        if self.mcp_client and self.mcp_client.initialized:
            try:
                res: ToolCallResult = await self.mcp_client.call_tool(tool_name, args, timeout=self.config.timeout_seconds)
                if res.is_error:
                    return f"Hubo un error al ejecutar la herramienta {tool_name}: {res.text}"
                return self.format_for_speech(res.text)
            except Exception as e:
                logger.error(f"[ExpertSupervisor] MCP tool execution error ({tool_name}): {e}", exc_info=True)
                return f"Error al ejecutar la herramienta {tool_name} vía MCP."

        # Fallback to general expert
        query = f"Ejecuta la acción {tool_name} con parámetros {args}"
        res = await self.execute(query)
        return res.get("text", "Completé la consulta.")


    @staticmethod
    def clean_query_text(text: str) -> str:
        """Strip introductory conversational phrases so the expert gets a clean prompt."""
        q = text.strip()
        tag_m = _DELEGATE_TAG_REGEX.search(q)
        if tag_m:
            return tag_m.group(1).strip()
        q = re.sub(r"^(sure|ok|okay|yes|yeah|claro|por supuesto|seguro|bien)[,\s]+", "", q, flags=re.IGNORECASE)
        q = re.sub(
            r"^(let\s+me\s+(look\s*up|check|search|investigate|find\s+out|research|consult|ask)|"
            r"i('ll|\s+will)\s+(look\s*up|check|search|investigate|find\s+out)|"
            r"d[eé]jame\s+(consultar|averiguar|buscar|investigar|preguntar)|"
            r"voy\s+a\s+(consultar|averiguar|buscar|investigar|preguntar)|"
            r"(un\s+segundo|un\s+momento|dame\s+un\s+segundo|dame\s+un\s+momento),?\s*(voy\s+a\s+|le\s+)?(consultar|preguntar|averiguar|pido)?|"
            r"espera\s+(un\s+momento,?\s+|un\s+segundo,?\s+)?(que\s+)?(le\s+pregunto|consulto|averiguo)?)\s*(con\s+el\s+|al?\s+)?(experto|supervisor|agy|claude|minimax|groq)?\s*(about\s+|sobre\s+|para\s+)?",
            "",
            q,
            flags=re.IGNORECASE
        )
        q = re.sub(r"\b(with\s+(the\s+)?(expert|supervisor|agy|claude|minimax|groq)|al?\s+experto|con\s+el\s+experto)\b", "", q, flags=re.IGNORECASE).strip()
        q = q.strip(".,;:?! ")
        return q or text.strip()

    def get_filler_phrase(self, lang: str = "es") -> str:
        """Returns a fast, natural filler phrase to speak while expert processes."""
        phrases = self.config.fillers_es if "es" in lang.lower() else self.config.fillers_en
        return random.choice(phrases)

    _META_PATTERNS = [
        r"(call|ask|consult|delegate|delayed|pass|use)\s+(to\s+)?(the\s+|an?\s+)?(expert|supervisor|agy|claude|minimax|groq)",
        r"call\s+the\s+expert\s+delegation",
        r"(llama|llamale|preguntale|pasa|pasale|usa|delega)\s+(al?\s+|un\s+)?(experto|supervisor|groq)",
        r"^(calculate|comput[eo]|calcul[aá])\s+(that|this|it|eso|esto)[\s.!?,]*$",
        r"^(can\s+you\s+)?(do|search|find|check|look\s*up)\s+(it|that|this)[\s.!?,]*$",
        r"^(okay\s+|ok\s+)?(can\s+you\s+)?do\s+it[\s.!?,]*$",
        r"^(puedes|podes|hacelo|buscalo|fijate|dale|averigualo)[\s.!?,]*$",
        r"\b(let\s+me|i('ll|\s+will)|i\s+can)\s+(search|look\s*up)\b",
        r"\bsearch\s+(it|for\s+it|for\s+you)\b",
        r"^where\s+(are|is)\s+(them|it|the\s+calculation)",
        r"^no\s+i\s+mean\s+",
    ]

    _SMALL_TALK_PATTERNS = [
        r"^(hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
        r"^(how\s+are\s+you|how\s+do\s+you\s+do|how\s+is\s+it\s+going)\b",
        r"^(really\s+good|doing\s+well|fine\s+thank\s+you|i'?m\s+good)\b",
        r"^(thanks|thank\s+you|ok|okay|bye|goodbye)\b",
    ]

    def resolve_expert_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        If query is a meta-delegation command ('call the expert', 'delegate to supervisor', 'calculate that'),
        synthesize the substantive topic from prior user turns so the expert answers the real question
        rather than saying 'transferring you now'.
        """
        is_meta = any(re.search(p, query, re.I) for p in self._META_PATTERNS)
        if is_meta and history:
            substantive = []
            for h in history:
                if h.get("role") == "user":
                    c = h.get("content", "").strip()
                    not_meta = not any(re.search(p, c, re.I) for p in self._META_PATTERNS)
                    not_small_talk = not any(re.search(p, c, re.I) for p in self._SMALL_TALK_PATTERNS)
                    if not_meta and not_small_talk and len(c.split()) >= 2:
                        substantive.append(c)
            if substantive:
                resolved = " | ".join(substantive)
                logger.info(f"[ExpertSupervisor] Meta-delegation '{query[:50]}' resolved to substantive topic: '{resolved[:100]}'")
                return resolved
        return query

    def get_agy_conversation_id(self, session_id: str) -> Optional[str]:
        """Returns the active AGY conversation ID for a voice session."""
        agy = self.providers.get(ExpertProvider.AGY.value)
        if hasattr(agy, "get_conversation_id"):
            return agy.get_conversation_id(session_id)
        return None

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        provider_override: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute query with expert and return result."""
        t0 = time.perf_counter()
        provider = self.get_provider(provider_override)
        provider_name = provider.name
        resolved_query = self.resolve_expert_query(query, history)

        try:
            exec_kwargs: Dict[str, Any] = {
                "query": resolved_query,
                "history": history,
                "system_prompt": self.config.expert_system_prompt,
                "timeout": self.config.timeout_seconds,
            }
            if session_id:
                exec_kwargs["session_id"] = session_id

            try:
                raw_result = await provider.execute(**exec_kwargs)
            except TypeError:
                # If provider doesn't accept session_id argument
                exec_kwargs.pop("session_id", None)
                raw_result = await provider.execute(**exec_kwargs)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            clean_speech = self.format_for_speech(raw_result)
            return {
                "success": True,
                "provider": provider_name,
                "text": clean_speech,
                "raw_text": raw_result,
                "elapsed_ms": round(elapsed_ms, 1),
                "error": None,
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[ExpertSupervisor] Execution error ({provider_name}): {e}", exc_info=True)
            return {
                "success": False,
                "provider": provider_name,
                "text": "Disculpa, tuve un problema al consultar al experto.",
                "raw_text": "",
                "elapsed_ms": round(elapsed_ms, 1),
                "error": str(e),
            }

    @staticmethod
    def format_for_speech(text: str) -> str:
        """Clean markdown, code markers, asterisks, bullet points for natural TTS."""
        if not text:
            return ""
        # Remove markdown headers #, ##
        t = re.sub(r"#+\s*", "", text)
        # Remove bold/italic **text** or *text*
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        t = re.sub(r"\*([^*]+)\*", r"\1", t)
        # Remove backticks
        t = re.sub(r"`([^`]+)`", r"\1", t)
        t = re.sub(r"```[a-zA-Z]*\n?|```", "", t)
        # Remove leading bullet points - or *
        t = re.sub(r"^\s*[-*•]\s*", "", t, flags=re.MULTILINE)
        # Consolidate spaces and newlines
        t = re.sub(r"\n+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # ── Passive Response Guardrail ──────────────────────────────────────────
    # Patterns that match empty/passive AI responses that don't actually help
    _PASSIVE_PATTERNS = [
        r"^(sure|ok(ay)?|alright|of course|certainly|sounds good|yeah|yes|great|got it|right)[\s,!.]*$",
        r"^(sure|ok(ay)?|alright|of course|certainly),?\s+(let'?s\s+(do|talk|discuss|get|try)|i\s+can\s+help|that'?s\s+(interesting|great|cool|nice))",
        r"^(okay|ok|sure|yeah),?\s+(let\s+me|i('ll|\s+will)|i\s+can)\s+(search|look|find|check)",
        r"^(bueno|dale|claro|si),?\s+(d[eé]jame|voy\s+a|puedo)\s+(buscar|averiguar|fijarme)",
        r"\b(let\s+me\s+search\s+for\s+you|i\s+can\s+search\s+for\s+you)\b",
        r"^(i'?m\s+afraid\s+i\s+can'?t|i\s+can'?t\s+do\s+that)",
        r"^(claro|por supuesto|de acuerdo|okey|seguro|bueno|dale|bien)[\s,!.]*$",
        r"^(claro|por supuesto|de acuerdo),?\s+(hablemos|vamos|hagamos)",
    ]
    _PASSIVE_RE = [re.compile(p, re.IGNORECASE) for p in _PASSIVE_PATTERNS]

    # Patterns indicating the user made a substantive request (not just small talk)
    _USER_SUBSTANTIVE_PATTERNS = [
        r"\b(help|develop|build|create|make|code|program|design|calculate|compute|search|find|explain|tell\s+me|show\s+me|can\s+you)\b",
        r"\b(ayud|desarroll|constru|cre[aá]|progra|diseñ|calcul|busc|explic|dime|muestr|pued[eo]s)\b",
        r"\b(website|app|application|project|system|tool|page|database|printer|impresora|specs|specifications|hardware|pants|pantalones|ropa|clothes|needle|aguja)\b",
        r"\b(expert|supervisor|agy|claude|minimax|groq|delegate|delega)\b",
        r"\b(call\s+the\s+expert|llam[aá]\s+al\s+experto|consult|delegat?e?|pedi[rl]e)\b",
    ]
    _USER_SUBSTANTIVE_RE = [re.compile(p, re.IGNORECASE) for p in _USER_SUBSTANTIVE_PATTERNS]

    def is_passive_response(self, ai_text: str) -> bool:
        """Check if AI text is an empty/passive acknowledgment that doesn't actually help."""
        clean = ai_text.strip().rstrip(".")
        if not clean:
            return True
        for pat in self._PASSIVE_RE:
            if pat.search(clean):
                return True
        words = clean.split()
        if len(words) > 15:
            return False  # Longer responses are probably substantive
        return False

    def extract_user_request_from_history(self, history: List[Dict[str, str]]) -> Optional[str]:
        """Get the most recent substantive user request from dialog history in chronological order."""
        if not history:
            return None
        # Look at last 4 user turns in chronological order
        user_turns = [h["content"] for h in history if h.get("role") == "user"][-4:]
        # Concatenate recent user turns for context
        combined = " ".join(user_turns)
        for pat in self._USER_SUBSTANTIVE_RE:
            if pat.search(combined):
                return combined.strip()
        return None

    def should_guardrail_delegate(self, ai_text: str, dialog_history: List[Dict[str, str]]) -> Tuple[bool, Optional[str]]:
        """
        Passive response guardrail: if AI gave an empty response but user asked for
        something substantive, return (True, query_for_expert).
        """
        if not self.config.enabled:
            return False, None
        if not self.is_passive_response(ai_text):
            return False, None
        user_request = self.extract_user_request_from_history(dialog_history)
        if not user_request:
            return False, None
        logger.info(f"[Guardrail] Passive AI response detected: '{ai_text[:50]}' — user requested: '{user_request[:80]}'")
        return True, user_request
