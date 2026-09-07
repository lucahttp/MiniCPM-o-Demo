"""
Unit and Integration tests for Slow Loop Brain (Groq/OpenAI compatible)
and AGY Persistent Session Management.
"""
import asyncio
import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.expert import ExpertSupervisor, ExpertConfig, ExpertProvider
from core.expert.providers.agy_provider import AgyExpertProvider
from core.expert.providers.openai_provider import OpenAIExpertProvider
from core.expert.providers.minimax_provider import MiniMaxExpertProvider


class TestSlowLoopBrainAndSessions(unittest.IsolatedAsyncioTestCase):

    async def test_groq_slow_loop_direct_query(self):
        """Test Groq as OpenAI-compatible Slow Loop Brain answering directly in < 1s."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        groq_prov = supervisor.providers.get("groq")
        self.assertIsNotNone(groq_prov)
        self.assertTrue(groq_prov.is_available())

        res = await supervisor.execute(
            query="Responde en 1 sola oracion: Cual es el planeta mas cercano al sol?",
            provider_override="groq"
        )
        print("Groq direct response:", res.get("text"), f"({res.get('elapsed_ms')}ms)")
        self.assertTrue(res.get("success"))
        self.assertIn("mercurio", res.get("text", "").lower())
        self.assertLess(res.get("elapsed_ms", 99999), 3000)

    async def test_groq_slow_loop_tool_call_calculator(self):
        """Test Groq calling the calculator tool via function calling."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        
        # Query that forces tool use
        res = await supervisor.execute(
            query="Usa la herramienta calculadora para calcular exactamente: 489 * 37",
            provider_override="groq"
        )
        print("Groq calculator tool response:", res.get("text"), f"({res.get('elapsed_ms')}ms)")
        self.assertTrue(res.get("success"))
        self.assertIn("18093", res.get("text", ""))

    async def test_minimax_provider_live(self):
        """Test MiniMax international API endpoint (https://api.minimax.io/v1)."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        mm_prov = supervisor.providers.get("minimax")
        self.assertIsNotNone(mm_prov)
        self.assertTrue(mm_prov.is_available())

        res = await supervisor.execute(
            query="Responde en 1 oracion breve: Para que sirve la memoria RAM?",
            provider_override="minimax"
        )
        print("MiniMax response:", res.get("text"), f"({res.get('elapsed_ms')}ms)")
        self.assertTrue(res.get("success"))
        self.assertTrue(len(res.get("text", "")) > 10)

    async def test_agy_session_continuity_across_turns(self):
        """Test AGY managing memory across multiple turns within the same voice session."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        session_id = "test_voice_session_albatros_999"

        # Turn 1: Give secret data
        res1 = await supervisor.execute(
            query="Anota y recuerda esto para la sesion: el codigo del servidor es CIPRES-44.",
            provider_override="agy",
            session_id=session_id
        )
        print("AGY Turn 1:", res1.get("text"))
        self.assertTrue(res1.get("success"))

        conv_id = supervisor.get_agy_conversation_id(session_id)
        print("AGY Conversation ID captured:", conv_id)
        self.assertIsNotNone(conv_id)

        # Turn 2: Query secret data using the same session_id (no prompt context passed)
        res2 = await supervisor.execute(
            query="Cual era el codigo del servidor que te acabo de dar?",
            provider_override="agy",
            session_id=session_id
        )
        print("AGY Turn 2:", res2.get("text"))
        self.assertTrue(res2.get("success"))
        self.assertIn("cipres", res2.get("text", "").lower())


if __name__ == "__main__":
    unittest.main()
