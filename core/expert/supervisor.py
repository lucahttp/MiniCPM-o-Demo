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
    # Spanish explicit delegation intent
    r"\b(d[eé]jame|voy\s+a|un\s+segundo,?\s+voy\s+a|espera\s+que)\s+(consultar|preguntar|averiguar|pedirle|buscar)\s+(con\s+el\s+|al?\s+)?(experto|supervisor|agy|claude|minimax)\b",
    r"\b(consult[aá](ndo|r[eé])?|pregunt[aá](ndo|r[eé])?)\s+(al?\s+|con\s+el\s+)?(experto|supervisor|agy|claude|minimax)\b",
    r"\b(preg[uú]ntale?\s+a|consult[aá](le)?\s+al?)\s+(agy|claude|minimax|experto)\b",
    r"\b(le\s+pregunto\s+al?\s+(experto|supervisor|agy|claude|minimax))\b",
    r"\b(consultando\s+al?\s+(experto|supervisor|agy|claude|minimax))\b",

    # English explicit delegation intent
    r"\b(let\s+me|i('ll|\s+will)|one\s+second,?\s+i'll)\s+(check|ask|consult|find\s+out|look\s*up)\s+(with\s+the\s+|the\s+)?(expert|supervisor|agy|claude|minimax)\b",
    r"\b(consulting|asking)\s+(with\s+)?(the\s+)?(expert|supervisor|agy|claude|minimax)\b",
    r"\b(i'm\s+asking|checking\s+with)\s+(the\s+)?(expert|supervisor|agy|claude|minimax)\b",
    r"\b(ask(ing)?|consult(ing)?)\s+(with\s+)?(the\s+)?(expert|supervisor|agy|claude|minimax)\b",
]

_DELEGATE_TAG_REGEX = re.compile(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", re.IGNORECASE)

class ExpertSupervisor:
    """Orchestrates hybrid voice delegation to specialized frontier agents (AGY, Claude, MiniMax)."""

    def __init__(self, config: Optional[ExpertConfig] = None):
        self.config = config or ExpertConfig()
        self.providers: Dict[str, BaseExpertProvider] = {
            ExpertProvider.AGY.value: AgyExpertProvider(self.config.agy_path),
            ExpertProvider.CLAUDE.value: ClaudeExpertProvider(self.claude_path_clean(self.config.claude_path)),
            ExpertProvider.MINIMAX.value: MiniMaxExpertProvider(
                api_key=self.config.minimax_api_key,
                group_id=self.config.minimax_group_id,
                model=self.config.minimax_model,
                api_base=self.config.minimax_api_base,
            ),
            ExpertProvider.OPENAI.value: OpenAIExpertProvider(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                api_base=self.config.openai_api_base or "https://api.openai.com/v1",
            ),
        }

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
            r"espera\s+(un\s+momento,?\s+|un\s+segundo,?\s+)?(que\s+)?(le\s+pregunto|consulto|averiguo)?)\s*(con\s+el\s+|al?\s+)?(experto|supervisor|agy|claude|minimax)?\s*(about\s+|sobre\s+|para\s+)?",
            "",
            q,
            flags=re.IGNORECASE
        )
        q = re.sub(r"\b(with\s+(the\s+)?expert|al?\s+experto|con\s+el\s+experto)\b", "", q, flags=re.IGNORECASE).strip()
        q = q.strip(".,;:?! ")
        return q or text.strip()

    def get_filler_phrase(self, lang: str = "es") -> str:
        """Returns a fast, natural filler phrase to speak while expert processes."""
        phrases = self.config.fillers_es if "es" in lang.lower() else self.config.fillers_en
        return random.choice(phrases)

    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        provider_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute query with expert and return result."""
        t0 = time.perf_counter()
        provider = self.get_provider(provider_override)
        provider_name = provider.name

        try:
            raw_result = await provider.execute(
                query=query,
                history=history,
                system_prompt=self.config.expert_system_prompt,
                timeout=self.config.timeout_seconds,
            )
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
