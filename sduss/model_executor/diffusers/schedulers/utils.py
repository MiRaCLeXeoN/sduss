class BaseSchedulerStates:

    def __init__(self) -> None:
        self.timestep_idx = 0
    

    def update_states_one_step(self):
        raise NotImplementedError
    

    def get_next_timestep(self):
        raise NotImplementedError
    

    def get_step_idx(self):
        raise NotImplementedError
    

    def log_status(self):
        raise NotImplementedError
    
    
    def to_device(self, device) -> None:
        raise NotImplementedError


    def to_dtype(self, dtype) -> None:
        raise NotImplementedError


    def to_numpy(self) -> None:
        raise NotImplementedError


    def to_tensor(self) -> None:
        raise NotImplementedError


class BatchSupportScheduler:

    def batch_set_timesteps(self):
        raise NotImplementedError
    
    def batch_scale_model_input(self):
        raise NotImplementedError
    
    def batch_step(self):
        raise NotImplementedError