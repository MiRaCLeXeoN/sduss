import importlib

from typing import Optional, Type, List

from torch import nn

# Architecture -> (module_name, class)
_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
}

class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        
        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"sduss.model_executor.models.{module_name}")
        return getattr(module, model_cls_name, None)
    
    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

__all__ = [
    "ModelRegistry",
]