from abc import ABC, abstractmethod

class BaseSamplingParams(ABC):
    """Sampling parameters for text generation.  """

    @abstractmethod
    def __repr__(self) -> str:
        """Literal representation of samling parameters."""
        pass

    @abstractmethod
    def __eq__(self, __value: object) -> bool:
        pass