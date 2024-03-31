class BatchSupportScheduler:

    def batch_set_timesteps(self):
        raise NotImplementedError
    
    def batch_scale_model_input(self):
        raise NotImplementedError
    
    def batch_step(self):
        raise NotImplementedError