
import contextlib

from typing import Type

import torch

from transformers import PretrainedConfig

from sduss.config import ModelConfig
from sduss.model_executor.weight_utils import initialize_dummy_weights

_MODEL_REGISTRY = {
    # ! NOT IMPLEMENTED YET
}

def _get_model_architecture(config: PretrainedConfig) -> Type[torch.nn.Module]:
    """Get the model architecture(nn.Module) based on PretrainedConfig."""
    architectures = getattr(config, "architectures", [])
    if not architectures:
        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {list(_MODEL_REGISTRY.keys())}"
        )
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]

@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Change the default torch dtype temporarily in the context."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

def get_model(model_config: ModelConfig) -> torch.nn.Module:
    # ? How to handle distributed situation?
    
    # model_class should be manually defined
    model_class = _get_model_architecture(model_config.hf_config)
    
    # ! Quantization check is omitted from the original version
    
    with _set_default_torch_dtype(model_config.dtype):
        model = model_class(model_config.hf_config)
        
        if model_config._load_format_is_dummy():
            model = model.cuda()
            initialize_dummy_weights(model)
        else:
            # load_weights method should be defined by model_class
            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
            model = model.cuda()
    return model.eval()