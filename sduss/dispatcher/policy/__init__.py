from .policy import DispatchPolicy
from .greedy import GreedyDispath

class DispatchPolicyFactory:
    
    # map: name -> (cls, support_mixed_precision)
    _POLICY_REGISTRY = {
        'greedy': (GreedyDispath, True),
    }
    
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> DispatchPolicy:
        # Check name
        if policy_name not in cls._POLICY_REGISTRY:
            raise ValueError(f"DispatchPolicy name {policy_name} is invalid. Please check up registry.")

        # Check mixed precision supportment
        use_mixed_precision = kwargs.pop("use_mixed_precision")
        if cls._POLICY_REGISTRY[policy_name][1] != use_mixed_precision:
            raise ValueError(f"Your designated policy {policy_name} is incompatible with {use_mixed_precision=}")

        return cls._POLICY_REGISTRY[policy_name][0](**kwargs)