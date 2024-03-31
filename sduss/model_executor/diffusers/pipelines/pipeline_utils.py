from typing import Dict
from dataclasses import fields

class BasePipeline:
    
    @classmethod
    def instantiate_pipeline(cls, **kwargs) -> "BasePipeline":
        raise NotImplementedError

class BasePipelineStepInput:
    
    def to_dict(self) -> Dict:
        """Convert to a dict. Only dataclass fields will be added into returned dict."""
        return {f.name:getattr(self, f.name) for f in fields(self)}


class BasePipelinePostInput:
    
    def to_dict(self) -> Dict:
        """Convert to a dict. Only dataclass fields will be added into returned dict."""
        return {f.name:getattr(self, f.name) for f in fields(self)}