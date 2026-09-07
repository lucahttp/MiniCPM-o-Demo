"""
core/expert/audio_gate.py

AudioStreamGate: Real-time audio and token stream gating for MiniCPM-o voice duplex.
Suppresses vocoder audio (Token2Wav / CosyVoice2) during model reasoning, native tool calling,
or technical syntax emission, manages verbal bridging, and unblocks natural voice upon expert response.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from .parser import fc2dict, resolve_ast_call
except ImportError:
    try:
        from core.expert.parser import fc2dict, resolve_ast_call
    except ImportError:
        fc2dict = None
        resolve_ast_call = None

logger = logging.getLogger(__name__)


class GateState(str, Enum):
    """
    States for the AudioStreamGate lifecycle:
    - SPEAKING: Audio waveform / tokens pass through normally to client and vocoder.
    - MUTED_TOOL_CALL: Audio is suppressed/silenced immediately when tool call, thought,
      or technical syntax is detected, preventing the vocoder from reading code/tags.
    - VERBAL_BRIDGING: Playing conversational filler phrase or earcon while waiting for expert/MCP.
    - UNMUTED_RESPONSE: Unblocks voice playback and prefill for the expert's clean response.
    """
    SPEAKING = "SPEAKING"
    MUTED_TOOL_CALL = "MUTED_TOOL_CALL"
    VERBAL_BRIDGING = "VERBAL_BRIDGING"
    UNMUTED_RESPONSE = "UNMUTED_RESPONSE"


@dataclass
class ToolCallEvent:
    """Represents an extracted native tool call or expert delegation query."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    is_expert: bool = False
    provider: Optional[str] = None
    query: Optional[str] = None


@dataclass
class GateChunkResult:
    """Result of processing a text/audio chunk through AudioStreamGate."""
    filtered_text: Optional[str]
    audio_data: Optional[Any]
    state: GateState
    tool_event: Optional[ToolCallEvent] = None
    is_muted: bool = False
    in_thought: bool = False
    earcon: Optional[str] = None


class AudioStreamGate:
    """
    Processes streaming text tokens and audio chunks in real-time.
    Detects whether the model is inside <|thought_start|> or <|tool_call_start|>,
    bracketed tags [TOOL: ...], or technical syntax.
    Controls audio muting and seamless state transitions.
    """

    # Primary token tags
    TAG_THOUGHT_START = "<|thought_start|>"
    TAG_THOUGHT_END = "<|thought_end|>"
    TAG_TOOL_START = "<|tool_call_start|>"
    TAG_TOOL_END = "<|tool_call_end|>"

    # Alternative / legacy XML tags
    TAG_ALT_THOUGHT_START = "<thought>"
    TAG_ALT_THOUGHT_END = "</thought>"
    TAG_ALT_TOOL_START = "<tool_call>"
    TAG_ALT_TOOL_END = "</tool_call>"

    # Regex patterns for complete blocks
    _THOUGHT_BLOCK_RE = re.compile(
        r"(?:<\|thought_start\|>|<thought>)(.*?)(?:<\|thought_end\|>|</thought>)",
        re.DOTALL | re.IGNORECASE,
    )
    _TOOL_BLOCK_RE = re.compile(
        r"(?:<\|tool_call_start\|>|<tool_call>)(.*?)(?:<\|tool_call_end\|>|</tool_call>)",
        re.DOTALL | re.IGNORECASE,
    )
    _BRACKET_TOOL_RE = re.compile(
        r"\[TOOL:\s*([a-zA-Z0-9_.-]+)\((.*?)\)\s*\]",
        re.DOTALL | re.IGNORECASE,
    )
    _BRACKET_EXPERT_RE = re.compile(
        r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]",
        re.DOTALL | re.IGNORECASE,
    )

    # Technical syntax patterns (e.g. code blocks)
    _CODE_BLOCK_START_RE = re.compile(r"```(?:json|python|bash|sh|javascript|js)?\s*", re.IGNORECASE)
    _CODE_BLOCK_END_RE = re.compile(r"```")

    # Prefix candidates that may indicate incoming tags across chunk boundaries
    _PREFIX_CANDIDATES = (
        "<|tool_call_start|>",
        "<|thought_start|>",
        "<tool_call>",
        "<thought>",
        "<function=",
        "[TOOL:",
        "[EXPERT:",
        "[DELEGATE:",
        "```",
    )

    def __init__(self, sample_rate: int = 16000, mute_as_silence: bool = False):
        self.sample_rate = sample_rate
        self.mute_as_silence = mute_as_silence
        self.state: GateState = GateState.SPEAKING

        # Internal streaming buffers
        self._text_buffer: str = ""
        self._tool_buffer: str = ""
        self._thought_buffer: str = ""
        self._in_tool_call: bool = False
        self._in_thought: bool = False
        self._in_code_block: bool = False
        self._last_filler: Optional[str] = None

    @property
    def is_muted(self) -> bool:
        """True if current state forbids vocal audio output."""
        return self.state in (GateState.MUTED_TOOL_CALL, GateState.VERBAL_BRIDGING)

    def mute_for_tool_call(self) -> None:
        """Explicitly transition to MUTED_TOOL_CALL state."""
        self.state = GateState.MUTED_TOOL_CALL
        logger.debug("[AudioStreamGate] State -> MUTED_TOOL_CALL")

    def start_verbal_bridging(self, filler_phrase: Optional[str] = None) -> None:
        """Transition to VERBAL_BRIDGING state playing filler or earcon."""
        self.state = GateState.VERBAL_BRIDGING
        self._last_filler = filler_phrase
        logger.debug(f"[AudioStreamGate] State -> VERBAL_BRIDGING (filler={filler_phrase})")

    def unmute_for_response(self, response_text: Optional[str] = None) -> None:
        """Transition to UNMUTED_RESPONSE to present expert answer."""
        self.state = GateState.UNMUTED_RESPONSE
        self._in_tool_call = False
        self._in_thought = False
        self._in_code_block = False
        self._tool_buffer = ""
        self._thought_buffer = ""
        logger.debug("[AudioStreamGate] State -> UNMUTED_RESPONSE")

    def on_prefill_injected(self) -> None:
        """Called when expert answer is prefills into model; resumes normal SPEAKING."""
        self.state = GateState.SPEAKING
        self._text_buffer = ""
        self._in_tool_call = False
        self._in_thought = False
        self._in_code_block = False
        logger.debug("[AudioStreamGate] State -> SPEAKING (post prefill injection)")

    def reset(self) -> None:
        """Reset gate state and buffers to default clean SPEAKING state."""
        self.state = GateState.SPEAKING
        self._text_buffer = ""
        self._tool_buffer = ""
        self._thought_buffer = ""
        self._in_tool_call = False
        self._in_thought = False
        self._in_code_block = False
        self._last_filler = None
        logger.debug("[AudioStreamGate] Reset to clean initial state")

    def _find_potential_prefix_start(self, text: str) -> Optional[int]:
        """
        Check if the tail of text matches a prefix of any trigger tag.
        Returns the start index of the prefix in text, or None.
        Used to prevent vocoder audio leaks on split tokens (e.g. '<|tool_').
        """
        if not text:
            return None
        text_lower = text.lower()
        earliest_match: Optional[int] = None
        for prefix in self._PREFIX_CANDIDATES:
            prefix_lower = prefix.lower()
            for i in range(2, len(prefix_lower)):
                sub = prefix_lower[:i]
                if text_lower.endswith(sub):
                    idx = len(text) - len(sub)
                    if earliest_match is None or idx < earliest_match:
                        earliest_match = idx
        return earliest_match

    @staticmethod
    def parse_tool_call_string(raw: str) -> Optional[ToolCallEvent]:
        """
        Parse raw tool call content into ToolCallEvent.
        Leverages MiniCPM native parser (fc2dict / resolve_ast_call) with robust fallbacks.
        Supports:
        1. OpenBMB native AST syntax (inside ```python ... ``` or direct func())
        2. JSON format: {"name": "calc", "arguments": {"expr": "2+2"}}
        3. Bracket format: [TOOL: name(args)]
        4. Expert query: [EXPERT: query] or [DELEGATE: query]
        """
        raw_clean = raw.strip()
        if not raw_clean:
            return None

        # 1. Use fc2dict if available
        if fc2dict is not None:
            test_str = raw_clean
            if not test_str.startswith("<|tool_call_start|>") and not test_str.startswith("["):
                test_str = f"<|tool_call_start|>\n```python\n{raw_clean}\n```\n<|tool_call_end|>"
            try:
                _, calls = fc2dict(test_str)
                if calls:
                    first = calls[0]
                    tool_name = first.get("name", "tool")
                    args = first.get("arguments", {})
                    is_exp = tool_name.lower() in ("expert", "delegate", "supervisor")
                    q = args.get("query") or args.get("__positional__")
                    return ToolCallEvent(
                        name=tool_name,
                        arguments=args,
                        raw=raw_clean,
                        is_expert=is_exp,
                        query=q,
                    )
            except Exception as e:
                logger.debug(f"[AudioStreamGate] fc2dict parse attempt failed: {e}")

        # 2. Check for [EXPERT: ...] or [DELEGATE: ...]
        m_exp = re.search(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", raw_clean, re.IGNORECASE)
        if m_exp:
            q = m_exp.group(1).strip()
            return ToolCallEvent(
                name="expert",
                arguments={"query": q},
                raw=raw_clean,
                is_expert=True,
                query=q,
            )

        # 3. Check for [TOOL: name(args)]
        m_brk = re.search(r"\[TOOL:\s*([a-zA-Z0-9_.-]+)\((.*?)\)\s*\]", raw_clean, re.IGNORECASE)
        if m_brk:
            tool_name = m_brk.group(1).strip()
            args_str = m_brk.group(2).strip()
            args = AudioStreamGate._parse_args_string(args_str)
            return ToolCallEvent(
                name=tool_name,
                arguments=args,
                raw=raw_clean,
                is_expert=(tool_name.lower() in ("expert", "delegate")),
                query=args.get("query") or args.get("__positional__"),
            )

        # 4. Check for JSON format
        if raw_clean.startswith("{") and raw_clean.endswith("}"):
            try:
                data = json.loads(raw_clean)
                tool_name = data.get("name") or data.get("tool") or data.get("function") or "tool"
                args = data.get("arguments") or data.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"__positional__": args}
                is_expert = tool_name.lower() in ("expert", "delegate", "supervisor")
                return ToolCallEvent(
                    name=tool_name,
                    arguments=args if isinstance(args, dict) else {"args": args},
                    raw=raw_clean,
                    is_expert=is_expert,
                    query=args.get("query") if isinstance(args, dict) else str(args),
                )
            except json.JSONDecodeError:
                pass

        # 5. AST call resolution
        if resolve_ast_call is not None:
            try:
                clean_expr = raw_clean
                m_fence = re.search(r"```(?:python)?\s*(.*?)\s*```", clean_expr, re.DOTALL | re.IGNORECASE)
                if m_fence:
                    clean_expr = m_fence.group(1).strip()
                res = resolve_ast_call(clean_expr)
                tool_name = res.get("name", "tool")
                args = res.get("arguments", {})
                is_exp = tool_name.lower() in ("expert", "delegate", "supervisor")
                return ToolCallEvent(
                    name=tool_name,
                    arguments=args,
                    raw=raw_clean,
                    is_expert=is_exp,
                    query=args.get("query") or args.get("__positional__"),
                )
            except Exception:
                pass

        # 6. Standard regex func_name(args)
        m_func = re.match(r"^([a-zA-Z0-9_.-]+)\s*\((.*)\)$", raw_clean, re.DOTALL)
        if m_func:
            tool_name = m_func.group(1).strip()
            args_str = m_func.group(2).strip()
            args = AudioStreamGate._parse_args_string(args_str)
            is_expert = tool_name.lower() in ("expert", "delegate", "supervisor")
            return ToolCallEvent(
                name=tool_name,
                arguments=args,
                raw=raw_clean,
                is_expert=is_expert,
                query=args.get("query") or args.get("__positional__"),
            )

        # Fallback: Treat as generic tool
        return ToolCallEvent(
            name="generic_tool",
            arguments={"query": raw_clean},
            raw=raw_clean,
            is_expert=False,
            query=raw_clean,
        )

    @staticmethod
    def _parse_args_string(args_str: str) -> Dict[str, Any]:
        """Parse keyword arguments like expr='2+2', location='Madrid' or positional string."""
        args: Dict[str, Any] = {}
        if not args_str:
            return args

        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))')
        matches = pattern.findall(args_str)
        if matches:
            for key, val_dq, val_sq, val_raw in matches:
                val = val_dq or val_sq or val_raw
                if val.isdigit():
                    args[key] = int(val)
                else:
                    try:
                        args[key] = float(val)
                    except ValueError:
                        args[key] = val
            return args

        stripped = args_str.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
            stripped = stripped[1:-1]
        return {"__positional__": stripped}

    def process_chunk(
        self,
        text: Optional[str] = None,
        audio_data: Optional[Any] = None,
    ) -> GateChunkResult:
        """
        Process a single streaming text/audio chunk.
        - Suppresses audio if entering/staying in MUTED_TOOL_CALL.
        - Extracts tool call event upon detection.
        - Filters out raw tool tags and technical reasoning from speech output.
        """
        text = text or ""
        tool_event: Optional[ToolCallEvent] = None
        earcon: Optional[str] = None
        output_text = ""

        # Update buffer
        self._text_buffer += text
        buf = self._text_buffer

        # Check for start of thought block
        if not self._in_thought:
            for s_tag in (self.TAG_THOUGHT_START, self.TAG_ALT_THOUGHT_START):
                if s_tag in buf:
                    self._in_thought = True
                    idx = buf.index(s_tag)
                    if self.state == GateState.SPEAKING:
                        output_text += buf[:idx]
                    self.state = GateState.MUTED_TOOL_CALL
                    self._text_buffer = buf[idx + len(s_tag):]
                    buf = self._text_buffer
                    logger.info("[AudioStreamGate] Detected thought start -> MUTED_TOOL_CALL")
                    break

        # If inside thought, look for thought end
        if self._in_thought:
            self.state = GateState.MUTED_TOOL_CALL
            for e_tag in (self.TAG_THOUGHT_END, self.TAG_ALT_THOUGHT_END):
                if e_tag in buf:
                    idx = buf.index(e_tag)
                    self._thought_buffer += buf[:idx]
                    self._in_thought = False
                    self._text_buffer = buf[idx + len(e_tag):]
                    buf = self._text_buffer
                    if not self._in_tool_call and not self._in_code_block and self.state == GateState.MUTED_TOOL_CALL:
                        self.state = GateState.SPEAKING
                    logger.info(f"[AudioStreamGate] Thought closed ({len(self._thought_buffer)} chars)")
                    break
            else:
                self._thought_buffer += buf
                self._text_buffer = ""
                buf = ""

        # Check for start of native tool call block
        if not self._in_tool_call and buf:
            for t_tag in (self.TAG_TOOL_START, self.TAG_ALT_TOOL_START):
                if t_tag in buf:
                    self._in_tool_call = True
                    idx = buf.index(t_tag)
                    if self.state == GateState.SPEAKING:
                        output_text += buf[:idx]
                    self.state = GateState.MUTED_TOOL_CALL
                    earcon = "tool_start"
                    self._text_buffer = buf[idx + len(t_tag):]
                    buf = self._text_buffer
                    logger.info("[AudioStreamGate] Detected native tool call start -> MUTED_TOOL_CALL")
                    break

        # If inside tool call block, accumulate and look for tool end
        if self._in_tool_call and buf:
            self.state = GateState.MUTED_TOOL_CALL
            for e_tag in (self.TAG_TOOL_END, self.TAG_ALT_TOOL_END):
                if e_tag in buf:
                    idx = buf.index(e_tag)
                    self._tool_buffer += buf[:idx]
                    self._in_tool_call = False
                    self._text_buffer = buf[idx + len(e_tag):]
                    buf = self._text_buffer
                    tool_event = self.parse_tool_call_string(self._tool_buffer)
                    logger.info(f"[AudioStreamGate] Native tool call completed: {tool_event.name if tool_event else 'unknown'}")
                    break
            else:
                self._tool_buffer += buf
                self._text_buffer = ""
                buf = ""

        # Check for code blocks ```python ... ```
        if not self._in_code_block and not self._in_tool_call and not self._in_thought and buf:
            m_code = self._CODE_BLOCK_START_RE.search(buf)
            if m_code:
                idx_s = m_code.start()
                if self.state == GateState.SPEAKING:
                    output_text += buf[:idx_s]
                self._in_code_block = True
                self.state = GateState.MUTED_TOOL_CALL
                self._text_buffer = buf[m_code.end():]
                buf = self._text_buffer
                logger.info("[AudioStreamGate] Code block detected -> MUTED_TOOL_CALL")

        if self._in_code_block and buf:
            self.state = GateState.MUTED_TOOL_CALL
            m_code_end = self._CODE_BLOCK_END_RE.search(buf)
            if m_code_end:
                self._in_code_block = False
                self._text_buffer = buf[m_code_end.end():]
                buf = self._text_buffer
                if not self._in_tool_call and not self._in_thought and self.state == GateState.MUTED_TOOL_CALL:
                    self.state = GateState.SPEAKING
                logger.info("[AudioStreamGate] Code block ended")
            else:
                self._text_buffer = ""
                buf = ""

        # Check for inline bracketed tools: [TOOL: ...] or [EXPERT: ...]
        if not tool_event and not self._in_tool_call and not self._in_thought and buf:
            m_tool = self._BRACKET_TOOL_RE.search(buf)
            if m_tool:
                idx_s = m_tool.start()
                idx_e = m_tool.end()
                if self.state == GateState.SPEAKING:
                    output_text += buf[:idx_s]
                self.state = GateState.MUTED_TOOL_CALL
                earcon = "tool_start"
                tool_event = self.parse_tool_call_string(m_tool.group(0))
                self._text_buffer = buf[idx_e:]
                buf = self._text_buffer
                logger.info(f"[AudioStreamGate] Bracket tool detected: {tool_event.name if tool_event else 'none'}")
            else:
                m_exp = self._BRACKET_EXPERT_RE.search(buf)
                if m_exp:
                    idx_s = m_exp.start()
                    idx_e = m_exp.end()
                    if self.state == GateState.SPEAKING:
                        output_text += buf[:idx_s]
                    self.state = GateState.MUTED_TOOL_CALL
                    earcon = "tool_start"
                    tool_event = self.parse_tool_call_string(m_exp.group(0))
                    self._text_buffer = buf[idx_e:]
                    buf = self._text_buffer
                    logger.info(f"[AudioStreamGate] Bracket expert detected: {tool_event.query[:50] if tool_event and tool_event.query else ''}")

        # Check for potential tag prefixes to prevent vocoder leaks on token boundaries
        if self.state == GateState.SPEAKING and not self._in_tool_call and not self._in_thought and not self._in_code_block and buf:
            p_idx = self._find_potential_prefix_start(buf)
            if p_idx is not None:
                output_text += buf[:p_idx]
                self._text_buffer = buf[p_idx:]
                self.state = GateState.MUTED_TOOL_CALL
            else:
                output_text += buf
                self._text_buffer = ""

        # Determine audio muting
        mute_audio = (
            self.state in (GateState.MUTED_TOOL_CALL, GateState.VERBAL_BRIDGING)
            or self._in_tool_call
            or self._in_thought
            or self._in_code_block
        )

        out_audio = None if mute_audio else audio_data

        if mute_audio and self.mute_as_silence and audio_data is not None:
            out_audio = self._generate_silence_like(audio_data)

        clean_text = output_text if output_text else None

        return GateChunkResult(
            filtered_text=clean_text,
            audio_data=out_audio,
            state=self.state,
            tool_event=tool_event,
            is_muted=mute_audio,
            in_thought=self._in_thought,
            earcon=earcon,
        )

    def _generate_silence_like(self, audio_data: Any) -> Any:
        """Create a silent placeholder matching the type and size of audio_data."""
        try:
            import numpy as np
            if isinstance(audio_data, np.ndarray):
                return np.zeros_like(audio_data)
            if isinstance(audio_data, (bytes, bytearray)):
                return b"\x00" * len(audio_data)
        except Exception:
            pass
        return None
