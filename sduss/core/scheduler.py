import enum
import time

from typing import List, Optional, Tuple, Dict

from sduss.config import SchedulerConfig, CacheConfig
from sduss.core.policy import PolicyFactory
from sduss.core.block_manager import BlockSpaceManager
from sduss.sequence import SequenceStatus, SequenceGroup, SequenceGroupMetadata
from sduss.logger import init_logger

logger = init_logger(__name__)

class PreemptionMode(enum.Enum):
    """Preemption Modes

    Attributes:
        SWAP: Swap out the blocks of the preempted sequences to CPU memory
            and swap the back in when the sequences are resumed.
        RECOMPUTE: Discard the blocks of the preempted sequences and recompute
            them when the sequences are resumed, treating the sequences as
            new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE  = enum.auto()

class SchedulerOutputs:    
    """Wrapper of scheduler output.
    
    Args:
        scheduled_seq_groups (List[SequenceGroup]): Scheduled to be run.
        prompt_run (bool): Whether the scheduled seq_groups are in prompt stage.
        num_batched_tokens (int): _description_
        blocks_to_swap_in (Dict[int, int]): _description_
        blocks_to_swap_out (Dict[int, int]): _description_
        blocks_to_copy (Dict[int, List[int]]): _description_
        ignored_seq_groups (List[SequenceGroup]): _description_
    """
    
    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups
    
    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)
        
class Scheduler:
    """Main scheduler which arranges tasks.
    
    Attributes:
        prompt_limit: Length limit of the prompt derived from configuration.
    """
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        
        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)
        
        # Scheduler policy
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
        )
        
        # Sequence groups in the WAITING state. Haven't started to run.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
    
    def _schedule(self) -> SchedulerOutputs:
        # ? Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        
        # Fix the current time
        now = time.monotonic()
        
        # Launch new waiting sequences if possible
        # Only when there is no task swapped out.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled_seq_groups: List[SequenceGroup] = []
            seq_lens: List[int] = []
            
            # The total number of sequences on the fly.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            
            # Iterate through all waiting SequenceGroups to find as many eligible
            # tasks to start in next round as possible.
            while self.waiting:
                seq_group = self.waiting[0] # seq_group to be handled in this iteration
                
                assert seq_group.num_seqs() == 1, (
                    "SequenceGroup in waiting status should have only one "
                    "prompt sequence at the time."
                )
                # If prompt length exceeds the limit, ignore this seq_group.
                num_prompt_tokens = seq_group.get_seqs()[0].get_prompt_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)  # Get rid of this seq_group
                    continue
                
                # If the seq_group cannot be allocated, stop finding more and break
                if not self.block_manager.can_allocate(seq_group):
                    break
                
                # If the number of batched tokens exceeds the limit, stop and break
                new_seq_lens = seq_lens + [num_prompt_tokens]  # Accumulate prompt length
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)  # Padded
                if num_batched_tokens > self.scheduler_config.max_num_batched_tokens:
                    break
                
                # The total number of sequences in the RUNNING state should not exceed
                # the maximum number of sequence defined in the configuration
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break
                
                # The total paddings should not exceed the limit
                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                
                # Now the seq_group is qualified to be scheduled
                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled_seq_groups.append(seq_group)
            
            # All seq_groups to be launched have been gathered
            if scheduled_seq_groups or ignored_seq_groups:
                return SchedulerOutputs(
                    scheduled_seq_groups=scheduled_seq_groups,
                    prompt_run=True,
                    num_batched_tokens=len(new_seq_lens) * max(new_seq_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,             
                )
        
        # If there are swapped out tasks
        self.running = self.policy.sort_by_priority(now, self.running)
        # Reserve new token slots for the running sequence groups
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups
                    victim_seq_group = self.running.pop(-1)
                    self.
            
        
                
    
                
            
    
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        """Schedule sequence groups.

        Returns:
            Tuple[List[SequenceGroupMetadata], SchedulerOutputs]: 
        """
        scheduler_outputs = self
        
        
    def _allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate space for a seq_group to run.
        
        SequenceStatus is changed to RUNNING.
        """
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING
            
    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        
        if preemption_mode == PreemptionMode.RECOMPUTE:
            pass
        
    
    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1, ("Preemption by recomputation supports only "
            "seq_groups with 1 running sequence.")
        
        for seq in seqs:
            # We treat them as new prompts.
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        
        self.waiting.insert(0, seq_group)
        
    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        pass
    
    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)