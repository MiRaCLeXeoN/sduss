import os
import filelock
import glob
import json
import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional, Iterator, Tuple

import torch

from safetensors.torch import load_file, save_file, safe_open
from transformers import PretrainedConfig
from huggingface_hub import snapshot_download

from sduss.model_executor.layers.quantization import QuantizationConfig, get_quantization_config

def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock

def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)
        
def _shared_pointers(state_dict)->List[List]:
    """Get the names of the tensors whose data are shared by more than one tensor.

    Args:
        state_dict (_type_): state_dict of the model in pytorch.

    Returns:
        List[List]: List of list of names
    """
    ptrs = defaultdict(list)
    for k, v in state_dict.items():
        ptrs[v.data_ptr()].append(k)
    
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing
    

def convert_bin_to_safetensor_file(
    pt_filename:str,
    sf_filename:str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    
    # ? Remove shared weights from the dict
    shared_tensor_names = _shared_pointers(loaded)
    for shared_weights in shared_tensor_names:
        for name in shared_weights[1:]:
            loaded.pop(name)
    
    # Make tensors to be contiguous
    loaded = {k:v.contiguous() for k, v in loaded.items()}
    
    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # Check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
            - {sf_filename}: {sf_size}
            - {pt_filename}: {pt_size}
            """)
    
    # Check if the tensors are the same
    reloaded_sf = load_file(sf_filename)
    for name in reloaded_sf:
        pt_tensor = loaded[name]
        sf_tensor = reloaded_sf[name]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The converted safetensor file does not match the original pt file")
    
    
# TODO: Move to another place
def get_quant_config(
    quantization:str,
    model_name_or_path:str,
    hf_config:PretrainedConfig,
    cache_dir:Optional[str] = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(quantization)
    # Get the hf quantization config if available
    hf_quant_config = getattr(hf_config, "quantization_config", None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    
def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight) 

def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]
        
    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        load_format=load_format,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)
    
    if load_format == "npcache":
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()
    