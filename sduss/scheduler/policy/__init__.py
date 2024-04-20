from .FCFS import FCFS

class PolicyFactory:
    
    _POLICY_REGISTRY = {
        'fcfs': FCFS,
    }
    
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs):
        return cls._POLICY_REGISTRY[policy_name](**kwargs)