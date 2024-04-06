from typing import Dict
from dataclasses import fields

class BasePipeline:
    
    @classmethod
    def instantiate_pipeline(cls, **kwargs) -> "BasePipeline":
        raise NotImplementedError

class BasePipelineStepInput:
    pass
    

class BasePipelinePostInput:
    pass
    