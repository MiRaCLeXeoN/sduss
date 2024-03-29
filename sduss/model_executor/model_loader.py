
import contextlib
import os
import importlib
import json

from typing import Type, Tuple, Dict, List

import torch

from transformers import PretrainedConfig

from sduss.config import PipelineConfig
from sduss.model_executor.weight_utils import initialize_dummy_weights, get_quant_config
from .diffusers.pipelines import PipelneRegistry

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


def load_module(pipeline_pth: str, json_dict: Dict[str, List]) -> Dict:
    """Load submodules needed to initialize a pipelien.

    Args:
        pipeline_pth (str): Pipeline folder path.
        json_dict (Dict[str, List]): json dictionary.

    Returns:
        Dict: init keyword arguments.
    """
    ret = {}
    pkg_name = __name__[:__name__.rfind(".")]
    for name, l in json_dict.items():
        if name.startswith("_") or not isinstance(l, list):
            continue
        
        module_name = l[0]
        class_name = l[1]
        import_pth = pkg_name + "." + module_name + "." + class_name
        cls = importlib.import_module(import_pth)

        ret[name] = cls.from_pretrained(pipeline_pth, subfolder=name)
    return ret


def get_pipeline(pipeline_config: PipelineConfig) -> torch.nn.Module:
    pipeline_pth = pipeline_config.pipeline
    if not os.path.isdir(pipeline_pth):
        raise RuntimeError("Currently we only support local pipelines (you should "
                           f"download your model manually first).")
    with open(pipeline_pth + "/model_index.json") as f:
        file = json.load(f)
        class_name = f._class_name
        path_tuple : Tuple[str, str] = PipelneRegistry.get(class_name, None)
        if path_tuple is None:
            raise RuntimeError(f"The pipeline of designated model {pipeline_pth} is not supported "
                            f"yet. Currently, only the following pipelines can be properly launched:"
                            f"{PipelneRegistry}")
        import_path = ".diffusers.pipelines." + path_tuple[0] + "." + path_tuple[1]
        pipeline_cls = importlib.import_module(import_path)
        



    # model_class should be manually defined
    model_class = _get_model_architecture(model_config)
    
    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.pipeline,
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
                model_config.pipeline,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
            model = model.cuda()
    return model.eval()