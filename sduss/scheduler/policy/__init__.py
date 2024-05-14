from .FCFS_Single import FCFS_Single
from .FCFS_Mixed import FCFS_Mixed
from .Orca_ResByRes import OrcaResByRes
from .Orca_Round_Robin import OrcaRoundRobin
from .ESyMReD import ESyMReD_Scheduler

from .policy import Policy

class PolicyFactory:
    
    # map: name -> (cls, support_mixed_precision)
    _POLICY_REGISTRY = {
        'fcfs_single': (FCFS_Single, False),
        'fcfs_mixed' : (FCFS_Mixed, True),
        'orca_resbyres' : (OrcaResByRes, False),
        'orca_round_robin' : (OrcaRoundRobin, False),
        'esymred' : (ESyMReD_Scheduler, True),
    }
    
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        # Check name
        if policy_name not in cls._POLICY_REGISTRY:
            raise ValueError(f"Policy name {policy_name} is invalid. Please check up registry.")
        # Check mixed precision supportment
        use_mixed_precision = kwargs.pop("use_mixed_precision")
        if cls._POLICY_REGISTRY[policy_name][1] != use_mixed_precision:
            raise ValueError(f"Your designated policy {policy_name} is incompatible with {use_mixed_precision=}")

        return cls._POLICY_REGISTRY[policy_name][0](**kwargs)