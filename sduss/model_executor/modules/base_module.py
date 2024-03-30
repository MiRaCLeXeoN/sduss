from torch import nn

class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module
    ):
        super(BaseModule, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError