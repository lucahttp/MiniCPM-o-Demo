import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ExpertProvider(str, Enum):
    AGY = "agy"
    CLAUDE = "claude"
    MINIMAX = "minimax"
    OPENAI = "openai"

class ExpertConfig(BaseModel):
    """Configuration for the GPT-Live / OpenAI Realtime style Expert Supervisor."""
    enabled: bool = Field(default=True, description="Enable expert supervisor delegation")
    provider: ExpertProvider = Field(default=ExpertProvider.AGY, description="Default expert provider")
    
    # CLI paths
    agy_path: str = Field(
        default=os.environ.get("AGY_PATH", r"C:\Users\lucas\AppData\Local\agy\bin\agy.exe" if os.path.exists(r"C:\Users\lucas\AppData\Local\agy\bin\agy.exe") else "agy"),
        description="Path to agy executable"
    )
    claude_path: str = Field(
        default=os.environ.get("CLAUDE_PATH", r"C:\Users\lucas\.local\bin\claude.exe" if os.path.exists(r"C:\Users\lucas\.local\bin\claude.exe") else "claude"),
        description="Path to claude executable"
    )
    
    # MiniMax API config
    minimax_api_key: Optional[str] = Field(default=os.environ.get("MINIMAX_API_KEY"), description="MiniMax API key")
    minimax_group_id: Optional[str] = Field(default=os.environ.get("MINIMAX_GROUP_ID"), description="MiniMax Group ID if needed")
    minimax_model: str = Field(default="MiniMax-Text-01", description="MiniMax model name")
    minimax_api_base: str = Field(default="https://api.minimax.chat/v1", description="MiniMax API base URL")

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
            "Déjame consultar eso con el experto...",
            "Dame un segundo, ya te lo averiguo...",
            "Un momento, estoy consultando esa información...",
            "Déjame pensar en eso...",
            "Estoy revisando los datos para ti...",
        ]
    )
    fillers_en: List[str] = Field(
        default=[
            "Let me check that with the expert...",
            "Give me a second, looking that up for you...",
            "One moment, checking that information now...",
            "Let me look into that...",
            "Consulting the knowledge base, just a moment...",
        ]
    )
