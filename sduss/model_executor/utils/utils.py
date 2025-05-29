import random

from typing import Optional, Any, Dict

import numpy as np
import torch

def set_random_seed(seed: int) -> None:
    """Set seed for random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for attr_name, attr in weight_attrs.items():
        assert not hasattr(weight, attr_name), (f"Overwriting existing "
            f"tensor attribute: {attr_name}")
        setattr(weight, attr_name, attr)