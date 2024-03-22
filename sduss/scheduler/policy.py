from typing import List

from sduss.sequence import SequenceGroup

class Policy:
    
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError(
            "Policy base class method get_priority is called. Please implement"
            "this method in the derivative class for any usage."
        )
    
    def sort_by_priority(
        self,
        now: float,
        seq_groups: SequenceGroup,
    ) -> List[SequenceGroup]:
        return sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )


class FCFS(Policy):
    """First Come First Serve."""
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        """Returns a float as a comparison metric for sorted.

        Args:
            now (float): Time at now.
            seq_group (SequenceGroup): A SequenceGroup

        Returns:
            float: Comparison result.
        """        
        return now - seq_group.arrival_time


class PolicyFactory:
    
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }
    
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)