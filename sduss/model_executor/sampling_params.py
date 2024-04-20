import copy

from dataclasses import dataclass
from typing import List, Optional 

import torch

from sduss.logger import init_logger

logger = init_logger(__name__)

class BaseSamplingParams():
    """Sampling parameters for text generation.  """

    def __init__(self, **kwargs) -> None:
        """You should define volatile paramemters here."""

        self.prompt: str = kwargs.pop("prompt", "")
        self.negative_prompt: Optional[str] = kwargs.pop("negative_prompt", "")
        self.resolution: int = kwargs.pop("resolution", 512)
        self.num_inference_steps: int = kwargs.pop("num_inference_steps", 50)
        self.latents: Optional[torch.FloatTensor] = kwargs.pop("latents", None)
        self.prompt_embeds: Optional[torch.FloatTensor] = kwargs.pop("prompt_embeds", None)
        self.negative_prompt_embeds: Optional[torch.FloatTensor] = kwargs.pop("negative_prompt_embeds", None)

        # Check lantents' device
        if self.latents is not None and self.latents.device != torch.device("cpu"):
            logger.info(f"Forcing input lantents from {self.latents.device} to cpu to make sure "
                        f"it can be properly passed to worker through Ray.")
            self.latents.to("cpu")
        
        # Embeds must be None
        assert self.prompt_embeds is None and self.negative_prompt_embeds is None, (
            "Currently we don't support customized embeds.")
    
    
    def is_compatible_with(self, sampling_params: "BaseSamplingParams") -> bool:
        """Whether this sampling params is compatible with another.

        Compatible means these two sampling params can be batched.

        We cannot determine whether these params influence the compatibility of two sampling
        params here, since different pipelines and scheduler will have different standard of
        compatibility.
        So we make a check as loose as possible here, leaving derived classes to determine.

        Args:
            sampling_params (BaseSamplingParams): Another sampling params.

        Returns:
            bool: Whether they are compatible.
        """
        return True


    def __repr__(self) -> str:
        """Literal representation of samling parameters."""
        raise NotImplementedError


    def clone(self) -> "BaseSamplingParams":
        return copy.deepcopy(self)
            

    def _check_volatile_params(self):
        """Check volatile params to ensure they are the same as default."""
        raise NotImplementedError("_check_volatile_params method must be overridden by derived classes.")