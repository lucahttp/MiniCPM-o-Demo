from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseExpertProvider(ABC):
    """Base class for all expert supervisor providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if dependencies/paths/keys for this provider exist."""
        pass

    @abstractmethod
    async def execute(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 20.0,
    ) -> str:
        """Execute query with expert and return spoken text answer."""
        pass
