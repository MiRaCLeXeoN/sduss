import os

from collections import defaultdict
from typing import Dict, List, Optional

import torch

from safetensors.torch import load_file, save_file, safe_open
from transformers import PretrainedConfig

from sduss.model_executor.layers.quantization import QuantizationConfig, get_quantization_config

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
    
    

    