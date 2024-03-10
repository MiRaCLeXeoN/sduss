
import contextlib

from typing import Type

import torch

from transformers import PretrainedConfig

from sduss.config import ModelConfig
from sduss.model_executor.weight_utils import initialize_dummy_weights, get_quant_config
from sduss.model_executor.models import ModelRegistry

def _get_model_architecture(config: PretrainedConfig) -> Type[torch.nn.Module]:
    """Get the model architecture(nn.Module) based on PretrainedConfig."""
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    
    raise ValueError(
        f"This model {architectures} is currently not supported yet."
    )

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
    
    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()
    
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