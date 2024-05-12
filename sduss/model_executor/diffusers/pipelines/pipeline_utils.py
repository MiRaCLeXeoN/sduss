from typing import Dict
from dataclasses import fields

class BasePipeline:
    SUPPORT_MIXED_PRECISION: bool = False
    
    @classmethod
    def instantiate_pipeline(cls, **kwargs) -> "BasePipeline":
        raise NotImplementedError
    
    @staticmethod
    def get_sampling_params_cls():
        raise NotImplementedError


class BasePipelinePrepareOutput:
    def to_device(self, device) -> None:
        raise NotImplementedError


class BasePipelineStepInput:
    pass
    

class BasePipelinePostInput:
    pass
    