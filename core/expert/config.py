import os
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


def _load_dotenv_if_present():
    """Lightweight .env loader without external dependencies."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    for env_path in [base_dir / ".env", Path(".env")]:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k not in os.environ and v:
                                os.environ[k] = v.strip("'\"")
            except Exception:
                pass


_load_dotenv_if_present()


class ExpertProvider(str, Enum):
    AGY = "agy"
    CLAUDE = "claude"
    MINIMAX = "minimax"
    OPENAI = "openai"
    GROQ = "groq"


class ExpertConfig(BaseModel):
    """Configuration for the GPT-Live / OpenAI Realtime style Expert Supervisor."""
    enabled: bool = Field(default=True, description="Enable expert supervisor delegation")
    provider: ExpertProvider = Field(
        default=ExpertProvider(os.environ.get("SLOW_LOOP_PROVIDER", "groq").lower())
        if os.environ.get("SLOW_LOOP_PROVIDER", "groq").lower() in [e.value for e in ExpertProvider]
        else ExpertProvider.AGY,
        description="Default expert provider"
    )
    
    # Slow Loop Brain (Universal OpenAI-compatible orchestrator)
    slow_loop_api_base: str = Field(
        default=os.environ.get("SLOW_LOOP_API_BASE", os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")),
        description="Base URL for OpenAI-compatible slow loop brain"
    )
    slow_loop_api_key: Optional[str] = Field(
        default=os.environ.get("SLOW_LOOP_API_KEY", os.environ.get("GROQ_API_KEY")),
        description="API key for slow loop brain"
    )
    slow_loop_model: str = Field(
        default=os.environ.get("SLOW_LOOP_MODEL", os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")),
        description="Model name for slow loop brain"
    )

    # CLI paths
    agy_path: str = Field(
        default=os.environ.get("AGY_PATH", r"C:\Users\lucas\AppData\Local\agy\bin\agy.exe" if os.path.exists(r"C:\Users\lucas\AppData\Local\agy\bin\agy.exe") else "agy"),
        description="Path to agy executable"
    )
    agy_default_project: str = Field(
        default=os.environ.get("AGY_DEFAULT_PROJECT", r"C:\Users\lucas\MiniCPM-o-Demo"),
        description="Default project folder for agy sessions"
    )
    claude_path: str = Field(
        default=os.environ.get("CLAUDE_PATH", r"C:\Users\lucas\.local\bin\claude.exe" if os.path.exists(r"C:\Users\lucas\.local\bin\claude.exe") else "claude"),
        description="Path to claude executable"
    )
    
    # MiniMax API config
    minimax_api_key: Optional[str] = Field(default=os.environ.get("MINIMAX_API_KEY"), description="MiniMax API key")
    minimax_group_id: Optional[str] = Field(default=os.environ.get("MINIMAX_GROUP_ID"), description="MiniMax Group ID if needed")
    minimax_model: str = Field(default=os.environ.get("MINIMAX_MODEL", "MiniMax-Text-01"), description="MiniMax model name")
    minimax_api_base: str = Field(default=os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io/v1"), description="MiniMax API base URL")

    # Groq API config
    groq_api_key: Optional[str] = Field(default=os.environ.get("GROQ_API_KEY"), description="Groq API key")
    groq_api_base: str = Field(default=os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1"), description="Groq API base URL")
    groq_model: str = Field(default=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), description="Groq model name")

    # OpenAI / Compatible API config
    openai_api_key: Optional[str] = Field(default=os.environ.get("OPENAI_API_KEY"), description="OpenAI API key")
    openai_api_base: Optional[str] = Field(default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")

    # General options
    timeout_seconds: float = Field(default=20.0, description="Max execution timeout in seconds")
    auto_delegate: bool = Field(default=False, description="Automatically delegate complex questions to the expert")
    
    # Speech-friendly voice prompt injected into expert
    expert_system_prompt: str = Field(
        default=(
            "You are a specialized expert assisting a real-time voice duplex agent. "
            "Answer the question directly, factually, and concisely. "
            "Do NOT use markdown headers, bold, bullet points, asterisks, or code blocks. "
            "Write in natural spoken conversational language in 1 to 3 short sentences so it can be spoken immediately."
        ),
        description="System prompt given to expert model to format response for voice"
    )

    # Conversational fillers while waiting for expert
    fillers_es: List[str] = Field(
        default=[
            "Déjame pensar... mmm...",
            "Estoy investigando sobre eso... mmm...",
            "A ver, déjame consultar los detalles... mmm...",
            "Dame un segundo, ya te lo averiguo... mmm...",
            "Déjame ver... mmm...",
        ]
    )
    fillers_en: List[str] = Field(
        default=[
            "Let me think... mmm...",
            "I'm doing some research about that... mmm...",
            "Let me look into that for you... mmm...",
            "Give me a moment, checking that... mmm...",
            "Let me see... mmm...",
        ]
    )
