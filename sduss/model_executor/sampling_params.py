from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional 

import torch

@dataclass
class BaseSamplingParams(ABC):
    """Sampling parameters for text generation.  """
    prompt: str = None
    negative_prompt: Optional[str] = None
    num_imgs: int = 1
    num_inference_steps: int = 50
    timesteps: List[int] = None
    latents: Optional[torch.FloatTensor] = None

    @abstractmethod
    def __repr__(self) -> str:
        """Literal representation of samling parameters."""
        pass

    @abstractmethod
    def __post_init__(self) -> None:
        """You should define volatile paramemters here."""
        pass

    @abstractmethod
    def is_compatible_with(self, sp: "BaseSamplingParams") -> bool:
        """Whether two requests' sampling parameters are compatible."""
        pass