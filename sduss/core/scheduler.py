import enum
import time

from typing import List, Optional, Tuple, Dict, Union, Iterable

from sduss.config import SchedulerConfig, CacheConfig
from sduss.core.policy import PolicyFactory
from sduss.core.block_manager import BlockSpaceManager
from sduss.sequence import (SequenceStatus, SequenceData, Sequence,
                            SequenceGroup, SequenceGroupMetadata)
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
        
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """Add a new sequence group to waiting queue."""
        self.waiting.append(seq_group)
        
    def abort_seq_group(self, request_ids: Union[str, Iterable[str]]) -> None:
        """Abort a handful of requests.

        Args:
            request_ids (Union[str, Iterable[str]]): Requests to be aborted.
        """        
        if isinstance(request_ids, str):
            request_ids = (request_ids, )  # transform into an iterable
        request_ids = set(request_ids)
        
        for state_queue in [self.running, self.waiting, self.swapped]:
            # To perform removal correctly, we have to iterate reversely.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if not seq.is_finished():
                            seq.status = SequenceStatus.FINISHED_ABORTED
                            self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:  # early exit
                        return
                            
    def free_seq(self, seq: Sequence) -> None:
        """Free one sequence's blocks."""
        self.block_manager.free(seq)
        
    def free_finished_seq_groups(self) -> None:
        # ! This may cause memory leak!
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]
    
    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)
    
    def _schedule(self) -> SchedulerOutputs:
        """Schedules running and swapping operations.

        Returns:
            SchedulerOutputs: All information generated by scheduling.
        """        
        # ? Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        
        # Fix the current time
        now = time.monotonic()
        
        # Launch only new waiting sequences if possible
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
            # ? Why only `scheduled` is returned? How about running ones?
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
        # Decides preempted and running seq_groups; allocate new slot;
        self.running = self.policy.sort_by_priority(now, self.running)
        # Reserve new token slots for the running sequence groups
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups
                    # ! This may somehow be sub-optimal. An overall once scanning
                    # ! can further simplify this procedure.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out, preemption_mode=None)
                    preempted.append(victim_seq_group)
                else:
                    # No other seq_group can be preempted.
                    # The current seq_group will be swapped out
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break  # No need to continue, no more targets to walk through
            else:
                # Preemption is decided, append slot for this seq_group
                # ? When to allocate new slots for preempted seq_groups?
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running
        
        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        # If there is preempted seq_group, there's no space for swapping-in.
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running)

            while self.swapped:
                seq_group = self.swapped[0]  # Get the highest priority one
                if not self.block_manager.can_swap_in(seq_group):
                    break
                
                # The total number of sequences in the RUNNING state should not
                # exceed the limit
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break
                
                # Can be swapped in
                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)  # running groups all require this
                num_curr_seqs += new_seq_lens
                self.running.append(seq_group)
                
        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs
                
            
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        """Schedule sequence groups.
        
        This call will change the status of seq_groups.

        Returns:
            Tuple[List[SequenceGroupMetadata], SchedulerOutputs]: Scheduled 
                sequence groups' metadata and scheduler's output.
            
        """
        scheduler_outputs = self._schedule()
        
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            physical_blocks_number: Dict[int, List[int]] = {}
            
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                physical_blocks_number[seq_id] = self.block_manager.get_block_table(seq)
            
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=physical_blocks_number,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        
        return seq_group_metadata_list, scheduler_outputs
        
    def _append_slot(
        self, 
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        """Append a slot(block) for a new token.
        
        A new token was added to logical blocks of this sequence in the last round.
        This helps to get a physical slot for this token.

        Args:
            seq_group (SequenceGroup): Target sequence group
            blocks_to_copy (Dict[int, List[int]]): A mapping of
                (src block number -> List[target block number(s)])
        """
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            
            # Some blocks need to be copied
            if ret is not None:
                src_block_number, dst_block_number = ret
                if src_block_number in blocks_to_copy.keys():
                    blocks_to_copy[src_block_number].append(dst_block_number)
                else:
                    blocks_to_copy[src_block_number] = [dst_block_number]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate physical memory blocks for a seq_group to run.
        
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
        """Preempt the seq_group.

        Args:
            seq_group (SequenceGroup): Target seq_group to be preempted.
            blocks_to_swap_out (Dict[int, int]): Information carrier.
                Mapping of swapped out blocks will be stored inside.
            preemption_mode (Optional[PreemptionMode], optional): Preemption mode. 
                If None, it will be assigned according to be situation. Recomputation
                is used by default. However, when the sequence group has multiple
                sequences, recomputation is not currently supported.
                Defaults to None.

        Raises:
            ValueError: _description_
        """        
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise ValueError("Invalid preemption mode.")
        
    
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
        
        # TODO(MX): This conforms only to FCFS, decoupling is required.
        self.waiting.insert(0, seq_group)
        
    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)
    
    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        """Swap in a sequence group.
        
        A sequence group is swapped in as a whole.
        This method is responsible for changing the sequence status.

        Args:
            seq_group (SequenceGroup): Target sequence group.
            blocks_to_swap_in (Dict[int, int]): Information carrier.
                Mapping (from_cpu_block number -> to_gpu_block number) will
                be updated into it.
        """        
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING
            
    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        """Swap out a sequence group
        
        A sequence group is swapped out as a whole.
        This method is responsible for changing the sequence status.

        Args:
            seq_group (SequenceGroup): Target sequence group
            blocks_to_swap_out (Dict[int, int]): Information carrier.
                Mapping (from_gpu_block number -> to_cpu_block number) will
                be updated into it.
        """
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
        