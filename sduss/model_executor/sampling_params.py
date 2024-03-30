import copy

from dataclasses import dataclass
from typing import List, Optional 

import torch

from sduss.logger import init_logger

logger = init_logger(__name__)

@dataclass
class BaseSamplingParams():
    """Sampling parameters for text generation.  """
    prompt: str = None
    negative_prompt: Optional[str] = None
    num_imgs: int = 1
    num_inference_steps: int = 50
    timesteps: List[int] = None
    latents: Optional[torch.FloatTensor] = None

    def __repr__(self) -> str:
        """Literal representation of samling parameters."""
        pass


    def __post_init__(self) -> None:
        """You should define volatile paramemters here."""
        # Check lantents' device
        if self.latents is not None and self.latents.device != torch.device("cpu"):
            logger.info(f"Forcing input lantents from {self.latents.device} to cpu for Ray compatibility.")
            self.latents.to("cpu")
        
        # TODO(MX): These custom configs are not supported yet
        assert self.timesteps is None, "We do not support custom timesteps at now."


    def clone(self) -> "BaseSamplingParams":
        return copy.deepcopy(self)
            

    def is_compatible_with(self, sp: "BaseSamplingParams") -> bool:
        """Whether two requests' sampling parameters are compatible."""
        pass