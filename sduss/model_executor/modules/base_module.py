from torch import nn

from diffusers import ConfigMixin, ModelMixin

class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module
    ):
        super(BaseModule, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class BaseModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        model: nn.Module
    ):
        super(BaseModel, self).__init__()
        self.model = model

    @property
    def config(self):
        return self.model.config
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError