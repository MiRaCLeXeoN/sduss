"""Main engine module.

Here defines the main base Engine class

"""

from typing import Optional, Union

from sduss.config import (ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig)

class LLMEngine:
    """The main engine that receives requests and generates texts.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_status: bool,
    ) -> None:
        """_summary_

        Args:
            model_config (ModelConfig): As name indicates
            cache_config (CacheConfig): As name indicates
            parallel_config (ParallelConfig): As name indicates
            scheduler_config (SchedulerConfig): As name indicates
            distributed_init_method (str): The initialization method for distributed
                execution. See `torch.distributed.init_process_group` for details.
            placement_group (Optional[PlacementGroup]): Ray placement group
                for distributed execution.
            log_status (bool): Whether to log statistics.
        """
        