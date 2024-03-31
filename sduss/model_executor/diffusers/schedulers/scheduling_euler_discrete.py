from diffusers import EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler

from .utils import BatchSupportScheduler

class EulerDiscreteSchedulerStates:
    """Scheduler states wrapper to store scheduler states of each request."""
    def __init__(self) -> None:
        pass

class EulerDiscreteScheduler(DiffusersEulerDiscreteScheduler, BatchSupportScheduler):
    def batch_set_timesteps(self):
        return super().batch_set_timesteps()
    
    def batch_scale_model_input(self):
        return super().batch_scale_model_input()
    
    def batch_step(self):
        return super().batch_step()