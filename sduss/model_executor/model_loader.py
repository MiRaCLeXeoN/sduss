
import contextlib
import os
import importlib
import json

from typing import Type, Tuple, Dict, List, Any

import torch

from transformers import PretrainedConfig

from .diffusers.pipelines import PipelneRegistry, EsyMReDPipelineRegistry

from sduss.config import PipelineConfig

PIPELINE_CLS = None

@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Change the default torch dtype temporarily in the context."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def load_modules(pipeline_pth: str, json_dict: Dict[str, List], kwargs: Dict) -> Dict:
    """Load submodules needed to initialize a pipelien.

    Args:
        pipeline_pth (str): Pipeline folder path.
        json_dict (Dict[str, List]): json dictionary.
        kwargs (Dict): kwargs for `from_pretrained`.

    Returns:
        Dict: init keyword arguments.
    """
    ret = {}
    pkg_name = __name__[:__name__.rfind(".")]
    for name, l in json_dict.items():
        if name.startswith("_") or not isinstance(l, list):
            continue
        
        module_name = l[0]
        if module_name == "stable_diffusion":
            module_name = "diffusers"
        class_name = l[1]
        import_pth = pkg_name + "." + module_name
        module = importlib.import_module(import_pth)
        cls = getattr(module, class_name)

        ret[name] = cls.from_pretrained(pipeline_pth, subfolder=name, **kwargs)
    return ret


def get_pipeline_cls(pipeline_config: PipelineConfig):
    global PIPELINE_CLS
    if PIPELINE_CLS is not None:
        return PIPELINE_CLS

    pipeline_pth = pipeline_config.pipeline
    if not os.path.isdir(pipeline_pth):
        raise RuntimeError("Currently we only support local pipelines (you should "
                           f"download your model to local first).")
    pipeline_registry = EsyMReDPipelineRegistry if pipeline_config.use_esymred else PipelneRegistry
    with open(pipeline_pth + "/model_index.json") as f:
        json_file = json.load(f)
        class_name = json_file["_class_name"]
        path_tuple : Tuple[str, str] = pipeline_registry.get(class_name, None)
        if path_tuple is None:
            raise RuntimeError(f"The pipeline of designated model {pipeline_pth} is not supported "
                            f"yet. Currently, only the following pipelines can be properly launched:"
                            f"{pipeline_registry}")
        import_path = ".model_executor.diffusers.pipelines." + path_tuple[0]
        module = importlib.import_module(import_path, package=__name__.split(".")[0])
        pipeline_cls = getattr(module, path_tuple[1])
        PIPELINE_CLS = pipeline_cls
    return pipeline_cls


def get_pipeline(pipeline_config: PipelineConfig, is_prepare_worker: bool = False) -> Any:
    pipeline_pth = pipeline_config.pipeline
    if not os.path.isdir(pipeline_pth):
        raise RuntimeError("Currently we only support local pipelines (you should "
                           f"download your model manually first).")
    
    pipeline_registry = EsyMReDPipelineRegistry if pipeline_config.use_esymred else PipelneRegistry
    with open(pipeline_pth + "/model_index.json") as f:
        json_file = json.load(f)
        class_name = json_file["_class_name"]
        path_tuple : Tuple[str, str] = pipeline_registry.get(class_name, None)
        if path_tuple is None:
            raise RuntimeError(f"The pipeline of designated model {pipeline_pth} is not supported "
                            f"yet. Currently, only the following pipelines can be properly launched:"
                            f"{pipeline_registry}")
        import_path = ".model_executor.diffusers.pipelines." + path_tuple[0]
        module = importlib.import_module(import_path, package=__name__.split(".")[0])
        pipeline_cls = getattr(module, path_tuple[1])

        global PIPELINE_CLS
        PIPELINE_CLS = pipeline_cls

        if is_prepare_worker:
            # We load the original huggingface pipeline, since prepare workers use only
            # the original features.
            # Since CPU device doesn't support float 16, we need to load float32    
            pipeline_config.kwargs["torch_dtype"] = torch.float32
            sub_modules = load_modules(pipeline_pth, json_file, pipeline_config.kwargs)
            pipeline = pipeline_cls.from_pretrained(pretrained_model_name_or_path=pipeline_pth, 
                                                     **sub_modules,
                                                     torch_dtype=torch.float32)
        else:
            sub_modules = load_modules(pipeline_pth, json_file, pipeline_config.kwargs)
            pipeline = pipeline_cls.instantiate_pipeline(pretrained_model_name_or_path=pipeline_pth, 
                                                        sub_modules=sub_modules,
                                                        **pipeline_config.kwargs)
    
    return pipeline