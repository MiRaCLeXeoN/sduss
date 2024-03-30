
import contextlib
import os
import importlib
import json

from typing import Type, Tuple, Dict, List, Any

import torch

from transformers import PretrainedConfig

from .diffusers.pipelines import PipelneRegistry, EsyMReDPipelineRegistry

from sduss.config import PipelineConfig


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
        import_pth = pkg_name + "." + module_name + "." + class_name
        cls = importlib.import_module(import_pth)

        ret[name] = cls.from_pretrained(pipeline_pth, subfolder=name, **kwargs)
    return ret


def get_pipeline(pipeline_config: PipelineConfig) -> Any:
    pipeline_pth = pipeline_config.pipeline
    if not os.path.isdir(pipeline_pth):
        raise RuntimeError("Currently we only support local pipelines (you should "
                           f"download your model manually first).")
    
    pipeline_registry = EsyMReDPipelineRegistry if pipeline_config.use_esymred else PipelneRegistry
    with open(pipeline_pth + "/model_index.json") as f:
        json_file = json.load(f)
        class_name = json_file._class_name
        path_tuple : Tuple[str, str] = pipeline_registry.get(class_name, None)
        if path_tuple is None:
            raise RuntimeError(f"The pipeline of designated model {pipeline_pth} is not supported "
                            f"yet. Currently, only the following pipelines can be properly launched:"
                            f"{pipeline_registry}")
        import_path = ".model_executor.diffusers.pipelines." + path_tuple[0] + "." + path_tuple[1]
        pipeline_cls = importlib.import_module(import_path, package=__name__.split(".")[0])

        sub_modules = load_modules(pipeline_pth, json_file, pipeline_config.kwargs)

        pipeline = pipeline_cls.instantiate_pipeline(pretrained_model_name_or_path=pipeline_pth, 
                                                     sub_modules=sub_modules,
                                                     **pipeline_config.kwargs)
    
    return pipeline