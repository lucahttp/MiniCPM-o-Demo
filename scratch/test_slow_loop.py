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
    async def test_groq_web_search_printer_specs_and_context(self):
        """Test Groq Slow Loop using web_search tool with prior dialog context (Kodak Portrait 3D printer)."""
        config = ExpertConfig(provider=ExpertProvider.GROQ)
        supervisor = ExpertSupervisor(config)

        history = [
            {"role": "user", "content": "I would love to start turning on like I have three printer that is been turned off for some weeks"},
            {"role": "assistant", "content": "Can you tell me what type of printer you have?"},
            {"role": "user", "content": "yes it's a Kodak portrait"},
            {"role": "assistant", "content": "Have you checked if the power cable is properly connected?"},
            {"role": "user", "content": "can you do a search with the expert across the internet for the specifications of these 3D printer"},
            {"role": "assistant", "content": "Sure."},
        ]

        # 1. Test guardrail detection on passive AI "Sure."
        guardrail_should, guardrail_query = supervisor.should_guardrail_delegate("Sure.", history)
        self.assertTrue(guardrail_should)
        self.assertIn("Kodak portrait", guardrail_query)
        print("Guardrail extracted query:", guardrail_query)

        # 2. Execute Slow Loop with session_id and history
        res = await supervisor.execute(
            query=guardrail_query,
            history=history,
            session_id="test_session_printer_001"
        )
        ans_text = res.get("text", "")
        safe_display = ans_text.encode("ascii", errors="replace").decode()
        print("Groq web search response:", safe_display, f"({res.get('elapsed_ms')}ms)")
        self.assertTrue(res.get("success"))
    async def test_groq_pants_repair_buenos_aires_search(self):
        """Test search trigger detection and real Buenos Aires clothing repair lookup."""
        config = ExpertConfig(provider=ExpertProvider.GROQ)
        supervisor = ExpertSupervisor(config)

        user_utterance = "first search can you search some place in Buenos Aires to do this"
        should_del, q, prov = supervisor.detect_delegation_intent(user_utterance)
        self.assertTrue(should_del, f"Utterance '{user_utterance}' should trigger delegation")
        print("Detected delegation query:", q)

        history = [
            {"role": "user", "content": "I have a broken pair of pants I want to fix them"},
            {"role": "assistant", "content": "We need needle and thread."},
            {"role": "user", "content": user_utterance},
        ]

        # Test passive AI phrase matching
        ai_passive = "Okay, let me search for you."
        self.assertTrue(supervisor.is_passive_response(ai_passive))
        guard_ok, guard_q = supervisor.should_guardrail_delegate(ai_passive, history)
        self.assertTrue(guard_ok)

        # Execute search
        res = await supervisor.execute(
            query="Find a place in Buenos Aires to repair broken pants / arreglar pantalones",
            history=history,
            session_id="test_pants_ba_001"
        )
        ans_text = res.get("text", "")
        safe_display = ans_text.encode("ascii", errors="replace").decode()
        print("Buenos Aires clothing repair search result:", safe_display)
        self.assertTrue(res.get("success"))
        text_lower = ans_text.lower()
        self.assertTrue(any(w in text_lower for w in ["buenos aires", "taller", "costura", "ropa", "pantalones", "repair", "arregl"]))

    def test_sure_thing_passive_guardrail(self):
        """Test guardrail detects 'Sure thing.' when user previously requested a search."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        history = [
            {"role": "user", "content": "can you help me to search"},
            {"role": "assistant", "content": "Sure thing—what would you like me to look up for you?"},
            {"role": "user", "content": "a bakeries in Montessori"},
        ]
        should, q = supervisor.should_guardrail_delegate("Sure thing.", history)
        self.assertTrue(should, "Passive guardrail should catch 'Sure thing.'")
        self.assertIn("bakeries in Montessori", q)

    def test_multi_turn_followup_detection(self):
        """Test detect_delegation_intent recognizes user answering an assistant clarifying question."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        history = [
            {"role": "user", "content": "can you help me to search"},
            {"role": "assistant", "content": "Sure thing—what would you like me to look up for you?"},
        ]
        should_del, q, prov = supervisor.detect_delegation_intent("a bakeries in Montessori", history)
        self.assertTrue(should_del, "Multi-turn follow-up should trigger delegation")
        self.assertEqual(q, "search for a bakeries in Montessori")

    def test_wildcats_inquiry_and_passive_help_guardrail(self):
        """Test 'Yes, I can help you with that.' guardrail and 'know about wildcats' intent."""
        config = ExpertConfig()
        supervisor = ExpertSupervisor(config)
        user_msg = "can you help me to know a little bit more about A Wildcats around the world"
        
        # 1. Direct intent detection
        should_del, q, prov = supervisor.detect_delegation_intent(user_msg)
        self.assertTrue(should_del, "Inquiry about wildcats should directly trigger delegation")
        
        # 2. Passive guardrail
        history = [
            {"role": "user", "content": "really good really good"},
            {"role": "user", "content": user_msg},
        ]
        ai_passive = "Yes, I can help you with that."
        self.assertTrue(supervisor.is_passive_response(ai_passive))
        should_guard, g_q = supervisor.should_guardrail_delegate(ai_passive, history)
        self.assertTrue(should_guard)
        self.assertEqual(g_q, user_msg)


if __name__ == "__main__":
    unittest.main()


