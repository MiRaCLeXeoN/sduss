
from typing import Optional, List, Dict, Tuple, Set

from sduss.utils import Device
from sduss.block import PhysicalTokenBlock
from sduss.sequence import Sequence, SequenceGroup, SequenceStatus

class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    
    Args:
        device (Device): CPU or GPU.
        block_size (int): Number of token IDs one block can hold.
        num_blocks (int): Number of available physical blocks in total.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:  
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(
                device=device,
                block_number=i,
                block_size=block_size
            )
            self.free_blocks.append(block)
    
    def allocate(self) -> PhysicalTokenBlock:
        """Allocate 1 physical block."""
        if not self.free_blocks:
            raise RuntimeError("Out of Memory! No free physical blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block
    
    def free(self, block: PhysicalTokenBlock) -> None:
        """Reduce 1 reference count.
        
        Only if reduced to 0, the block will be added back to free block list.
        Therefore, this is not actually the usual 'free' operation we have.
        """
        if block.ref_count == 0:
            raise ValueError(f"The block {block} is already freed!")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
    
    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)
    
# Mapping: logical block number -> physical block
BlockTable = List[PhysicalTokenBlock]

class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks.
    
    The instance of this class won't do anything real on physical memory.
    Every operation, like allocate and free, are performed logically.
    
    Args:
        block_size (int): Number of token IDs one block can hold.
        num_gpu_blocks (int): _description_
        num_cpu_blocks (int): _description_
        watermark (float, optional): The proportion of blocks attached with 
            a watermark to avoid frequent eviction.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): Size of the sliding window.
            Defaults to None.
    """
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:   
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        self.block_sliding_window = sliding_window
        
        if self.block_sliding_window is not None:
            assert self.block_sliding_window % block_size == 0, (sliding_window, block_size)
            self.block_sliding_window = self.block_sliding_window // block_size
        assert watermark >= 0.

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        
        # Mapping: seq_id -> BlockTable
        self.block_tables: Dict[int, BlockTable] = {}
    
    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME: Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs()[0]
        
        num_required_blocks = len(seq.logical_token_blocks)
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks, self.block_sliding_window)
        
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        
        # Use watermark to avoid frequent cache eviction
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks
        
    def allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate Physical blocks for prompt only.
        
        This method should be called at initialization stage to allocate
        blocks for prompt. Should not be used to allocate blocks for outputs.
        This method is not responsible for altering the status of sequences.
        """
        seq = seq_group.get_seqs()[0]
        
        block_table: BlockTable = []
        for logical_block_idx in range(len(seq.logical_token_blocks)):
            if (self.block_sliding_window is not None
                and logical_block_idx >= self.block_sliding_window):
                # Reuse the blocks inside sliding window
                block = block_table[logical_block_idx % self.block_sliding_window]
            else:
                # Allocate physical blocks for [0:self.block_sliding_window]
                block = self.gpu_allocator.allocate()
            block.ref_count = seq_group.num_seqs() # All sequences share the same prompt
            block_table.append(block)
        
        # Build block tables for each sequence
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()
        # ? Why is every block table the same?
    
    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        """Whether a new slot can be appended to every sequence in the group."""
        num_free_gpu_block = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_block
    
    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token.

        Args:
            seq (Sequence): _description_

        Returns:
            Optional[Tuple[int, int]]: _description_
        """
        # FIXME(MX): I cannot understand!
        raise RuntimeError("Method not understood.")
    
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]
        
        #
        if len(block_table) < len(logical_blocks):
            if (self.block_sliding_window is not None
                and len(block_table) >= self.block_sliding_window):
                # If we have to reuse a block, append one block reference to the end
                block_table.append(block_table[len(block_table) % self.block_sliding_window])
            else:
                #
                pass
        
        # Append the token to the last physical block
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. We can append.
            # ? Directly return???
            return None
        else:
            # The block is shared with other sequences.
            # Copy and write
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number
        
    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        """Whether a sequence group can be swapped in."""
        blocks = self._get_physical_blocks(seq_group)
        
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks
    
    def free(self, seq: Sequence) -> None:
        """Free blocks of one sequence."""
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet
            return
        self._free_block_table(self.block_tables[seq.seq_id])
        del self.block_tables[seq.seq_id]
    
    def _free_block_table(self, block_table: BlockTable) -> None:
        """Free all blocks one by one inside the block table."""
        for block in block_table:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)
    
    def _get_physical_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> List[PhysicalTokenBlock]:
        """Get all physical blocks of this sequence group.
        
        We assume that physical blocks are only shared by the sequences in the
        same group.
        Returned list has no repetition.
        """
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()  # Eliminate redundancy
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            else:
                blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)
        
        