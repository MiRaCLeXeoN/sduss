import torch
import torch.distributed

# Tensor model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Pipeline model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# A list of global ranks for each pipeline group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """Initialize model parallel groups
    
    Build tensor and pipeline parallel group via torch.distributed

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    """
    # ? why gpus are divided int groups like this?
    
    # make sure `torch.distributed.init_process_group` has been called
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    
    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size)
    rank = torch.distributed.get_rank()
    
    # Build the tensor parallel groups
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * pipeline_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
    
    # Build the pipeline parallel group
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline parallel group is already initialized"
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_GLOBAL_RANKS = ranks
            _PIPELINE_MODEL_PARALLEL_GROUP = group
        
    