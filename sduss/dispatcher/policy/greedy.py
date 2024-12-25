from .policy import DispatchPolicy

class GreedyDispath(DispatchPolicy):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)