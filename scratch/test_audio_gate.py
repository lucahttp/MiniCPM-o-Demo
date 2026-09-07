"""
scratch/test_audio_gate.py

Comprehensive test suite for AudioStreamGate and its integration with ExpertSupervisor and Worker.
Tests:
1. Four-state lifecycle: SPEAKING, MUTED_TOOL_CALL, VERBAL_BRIDGING, UNMUTED_RESPONSE.
2. Real-time token streaming & boundary buffering (preventing audio leaks on split tokens).
3. Thought block detection and audio muting (<|thought_start|>, <thought>).
4. Native tool call detection, parsing and audio muting (<|tool_call_start|>, <tool_call>).
5. Technical syntax detection (code blocks ```python, json payloads).
6. Legacy bracket tags ([TOOL:...], [EXPERT:...], [DELEGATE:...]).
7. Verbal bridging filler phrases and earcon signaling.
8. Unmuting upon expert response and prefill injection resumption.
9. Barge-in / interruption reset behavior.
10. ExpertSupervisor integration (detect_tool_or_expert, execute_tool_async).
11. Worker & C++ backend contract integrity (verifying worker.py imports and types).
"""

import os
import sys
import unittest
import numpy as np

# Ensure MiniCPM-o-Demo directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.expert.audio_gate import (
    AudioStreamGate,
    GateState,
    ToolCallEvent,
    GateChunkResult,
)
from core.expert.supervisor import ExpertSupervisor
from core.expert.config import ExpertConfig, ExpertProvider


class TestAudioStreamGate(unittest.TestCase):

    def setUp(self):
        self.gate = AudioStreamGate(sample_rate=16000)

    def test_initial_state(self):
        """Initial state should be SPEAKING and unmuted."""
        self.assertEqual(self.gate.state, GateState.SPEAKING)
        self.assertFalse(self.gate.is_muted)

    def test_normal_speaking_chunk(self):
        """Plain speech tokens and audio pass through unmodified."""
        dummy_audio = b"\x01\x02\x03\x04"
        res = self.gate.process_chunk("Hola, como estas hoy?", dummy_audio)
        self.assertEqual(res.state, GateState.SPEAKING)
        self.assertFalse(res.is_muted)
        self.assertEqual(res.filtered_text, "Hola, como estas hoy?")
        self.assertEqual(res.audio_data, dummy_audio)
        self.assertIsNone(res.tool_event)

    def test_thought_block_muting(self):
        """Thoughts delimited by <|thought_start|> should mute audio and strip text."""
        dummy_audio = b"\x05\x06\x07\x08"

        # Start of thought
        r1 = self.gate.process_chunk("Voy a pensar. <|thought_start|> analizo base de datos", dummy_audio)
        self.assertEqual(r1.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(r1.is_muted)
        self.assertIsNone(r1.audio_data)
        self.assertEqual(r1.filtered_text, "Voy a pensar. ")

        # Middle of thought
        r2 = self.gate.process_chunk(" y buscando registros", dummy_audio)
        self.assertEqual(r2.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(r2.is_muted)
        self.assertIsNone(r2.audio_data)
        self.assertIsNone(r2.filtered_text)

        # End of thought
        r3 = self.gate.process_chunk("<|thought_end|> Listo.", dummy_audio)
        self.assertEqual(r3.state, GateState.SPEAKING)
        self.assertFalse(r3.is_muted)
        self.assertEqual(r3.filtered_text, " Listo.")
        self.assertEqual(r3.audio_data, dummy_audio)

    def test_native_tool_call_muting_and_extraction(self):
        """Native <|tool_call_start|> blocks must silence audio and yield ToolCallEvent."""
        dummy_audio = "base64_audio_payload"

        # Token 1: Pre-tool text + tool call start
        r1 = self.gate.process_chunk("Un segundo. <|tool_call_start|>", dummy_audio)
        self.assertEqual(r1.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(r1.is_muted)
        self.assertIsNone(r1.audio_data)
        self.assertEqual(r1.filtered_text, "Un segundo. ")
        self.assertEqual(r1.earcon, "tool_start")

        # Token 2: Python tool call syntax inside markdown
        r2 = self.gate.process_chunk("```python\ncalculator(expr='25 * 4')\n```", dummy_audio)
        self.assertEqual(r2.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(r2.is_muted)
        self.assertIsNone(r2.audio_data)

        # Token 3: Tool call end
        r3 = self.gate.process_chunk("<|tool_call_end|>", dummy_audio)
        self.assertEqual(r3.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(r3.is_muted)
        self.assertIsNone(r3.audio_data)
        self.assertIsNotNone(r3.tool_event)
        self.assertEqual(r3.tool_event.name, "calculator")
        self.assertEqual(r3.tool_event.arguments, {"expr": "25 * 4"})

    def test_split_tag_streaming_leak_prevention(self):
        """Tokens split directly across chunk boundaries must be buffered to avoid audio leaks."""
        dummy_audio = "dummy_leak_check"

        # Chunk 1 splits tag prefix: "<|tool_"
        r1 = self.gate.process_chunk("Let me run that: <|tool_", dummy_audio)
        # Gate recognizes candidate prefix and suppresses audio
        self.assertEqual(r1.state, GateState.MUTED_TOOL_CALL)
        self.assertIsNone(r1.audio_data)
        self.assertEqual(r1.filtered_text, "Let me run that: ")

        # Chunk 2 completes start tag: "call_start|> time_date()"
        r2 = self.gate.process_chunk("call_start|> time_date()", dummy_audio)
        self.assertEqual(r2.state, GateState.MUTED_TOOL_CALL)
        self.assertIsNone(r2.audio_data)

        # Chunk 3 closes tag: "<|tool_call_end|>"
        r3 = self.gate.process_chunk("<|tool_call_end|>", dummy_audio)
        self.assertEqual(r3.state, GateState.MUTED_TOOL_CALL)
        self.assertIsNone(r3.audio_data)
        self.assertIsNotNone(r3.tool_event)
        self.assertEqual(r3.tool_event.name, "time_date")

    def test_verbal_bridging_lifecycle(self):
        """Verbal bridging state transitions and response unmuting."""
        self.gate.mute_for_tool_call()
        self.assertEqual(self.gate.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(self.gate.is_muted)

        # Start verbal bridging
        filler = "Déjame consultar al experto..."
        self.gate.start_verbal_bridging(filler)
        self.assertEqual(self.gate.state, GateState.VERBAL_BRIDGING)
        self.assertTrue(self.gate.is_muted)

        # Expert completes
        expert_answer = "La velocidad de la luz es aproximadamente 300.000 km/s."
        self.gate.unmute_for_response(expert_answer)
        self.assertEqual(self.gate.state, GateState.UNMUTED_RESPONSE)
        self.assertFalse(self.gate.is_muted)

        # Prefill is injected in C++ and speaking resumes
        self.gate.on_prefill_injected()
        self.assertEqual(self.gate.state, GateState.SPEAKING)
        self.assertFalse(self.gate.is_muted)

        # Subsequent output chunk passes through
        r = self.gate.process_chunk(expert_answer, "expert_wav_bytes")
        self.assertEqual(r.filtered_text, expert_answer)
        self.assertEqual(r.audio_data, "expert_wav_bytes")

    def test_legacy_bracket_tool_muting(self):
        """Bracket syntax [TOOL: calc(expr='3+3')] is detected, muted, and extracted."""
        res = self.gate.process_chunk("[TOOL: calculator(expr='3+3')]", "bracket_audio")
        self.assertEqual(res.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(res.is_muted)
        self.assertIsNone(res.audio_data)
        self.assertIsNotNone(res.tool_event)
        self.assertEqual(res.tool_event.name, "calculator")
        self.assertEqual(res.tool_event.arguments, {"expr": "3+3"})

    def test_legacy_bracket_expert_muting(self):
        """Bracket syntax [EXPERT: query] is detected as expert delegation and muted."""
        res = self.gate.process_chunk("[EXPERT: Explica la teoría de cuerdas]", "exp_audio")
        self.assertEqual(res.state, GateState.MUTED_TOOL_CALL)
        self.assertTrue(res.is_muted)
        self.assertIsNone(res.audio_data)
        self.assertIsNotNone(res.tool_event)
        self.assertTrue(res.tool_event.is_expert)
        self.assertEqual(res.tool_event.query, "Explica la teoría de cuerdas")

    def test_json_tool_call_parsing(self):
        """JSON function calling payload within <|tool_call_start|>."""
        json_call = '<|tool_call_start|>{"name": "fetch_weather", "arguments": {"city": "Paris", "temp": "C"}}<|tool_call_end|>'
        res = self.gate.process_chunk(json_call, "json_audio")
        self.assertEqual(res.state, GateState.MUTED_TOOL_CALL)
        self.assertIsNone(res.audio_data)
        self.assertIsNotNone(res.tool_event)
        self.assertEqual(res.tool_event.name, "fetch_weather")
        self.assertEqual(res.tool_event.arguments, {"city": "Paris", "temp": "C"})

    def test_mute_as_silence_option(self):
        """When mute_as_silence is True, returns zeroed array instead of None."""
        gate_silence = AudioStreamGate(sample_rate=16000, mute_as_silence=True)
        arr = np.ones((1600,), dtype=np.float32)

        res = gate_silence.process_chunk("<|tool_call_start|> calc() <|tool_call_end|>", arr)
        self.assertTrue(res.is_muted)
        self.assertIsNotNone(res.audio_data)
        self.assertTrue(np.all(res.audio_data == 0.0))
        self.assertEqual(len(res.audio_data), 1600)

    def test_barge_in_reset(self):
        """Interruption / barge-in resets gate back to clean SPEAKING state."""
        self.gate.process_chunk("<|thought_start|> deep thought", "thought_audio")
        self.assertEqual(self.gate.state, GateState.MUTED_TOOL_CALL)

        # Barge-in happens
        self.gate.reset()
        self.assertEqual(self.gate.state, GateState.SPEAKING)
        self.assertFalse(self.gate.is_muted)

        # Next normal turn speaks fine
        r = self.gate.process_chunk("Entendido.", "audio_ok")
        self.assertEqual(r.filtered_text, "Entendido.")
        self.assertEqual(r.audio_data, "audio_ok")


class TestSupervisorIntegration(unittest.TestCase):

    def setUp(self):
        self.supervisor = ExpertSupervisor()

    def test_supervisor_detect_tool_or_expert_native(self):
        """Supervisor detects OpenBMB native tool calls via parser."""
        text = "<|tool_call_start|>```python\ncalculator(expr='100 / 4')\n```<|tool_call_end|>"
        t_type, t_name, t_args = self.supervisor.detect_tool_or_expert(text)
        self.assertEqual(t_type, "TOOL")
        self.assertEqual(t_name, "calculator")
        self.assertEqual(t_args, {"expr": "100 / 4"})

    def test_supervisor_detect_bracket_tool(self):
        """Supervisor detects bracket tool calls."""
        text = "[TOOL: time_date(location='Tokyo')]"
        t_type, t_name, t_args = self.supervisor.detect_tool_or_expert(text)
        self.assertEqual(t_type, "TOOL")
        self.assertEqual(t_name, "time_date")
        self.assertEqual(t_args, {"location": "Tokyo"})

    def test_supervisor_execute_tool_async(self):
        """Supervisor executes instant tool asynchronously and returns clean speech."""
        import asyncio
        ans = asyncio.run(self.supervisor.execute_tool_async("calculator", {"expr": "7 * 8"}))
        self.assertIn("56", ans)

    def test_supervisor_create_audio_gate(self):
        """Supervisor factory method produces AudioStreamGate."""
        gate = self.supervisor.create_audio_gate()
        self.assertIsInstance(gate, AudioStreamGate)
        self.assertEqual(gate.state, GateState.SPEAKING)


class TestWorkerContractIntegrity(unittest.TestCase):

    def test_worker_imports_and_schemas(self):
        """Ensure worker.py and all related schemas and C++ backend definitions load without error."""
        from core.schemas.duplex import DuplexGenerateResult
        from core.schemas.streaming import StreamingChunk
        from core.expert import AudioStreamGate, ExpertSupervisor

        # Verify DuplexGenerateResult contract
        res = DuplexGenerateResult(
            is_listen=False,
            text="Test text",
            audio_data="sample_base64",
            current_time=1,
            cost_all_ms=10.0,
        )
        self.assertFalse(res.is_listen)
        self.assertEqual(res.text, "Test text")
        self.assertEqual(res.audio_data, "sample_base64")

        # Test gating on result
        gate = AudioStreamGate()
        chunk_res = gate.process_chunk("<|tool_call_start|> calculator(expr='2+2') <|tool_call_end|>", res.audio_data)
        self.assertTrue(chunk_res.is_muted)
        self.assertIsNone(chunk_res.audio_data)
        self.assertIsNotNone(chunk_res.tool_event)


if __name__ == "__main__":
    unittest.main(verbosity=2)
