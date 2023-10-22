import enum
import copy

from typing import List, Optional, Dict

from sduss.block import LogicalTokenBlock
from sduss.sampling_params import SamplingParams

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]

class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()  # Haven't started to run yet
    RUNNING = enum.auto()
    SWAPPED = enum.auto()  # Swapped out
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    
    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_STOPPED,
        ]
    
    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Optional[str]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a Sequence.
    
    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """
    
    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        
        # To be filled later
        # ? Why do we need cumulative log probability?
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0
        
    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.cumulative_logprob += logprob
        self.output_token_ids.append(token_id)
        
    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")

class Sequence:
    """All data belonging to a sequence.
    
    Attributes:
        seq_id: The unique id of the sequence.
        prompt: Prompt input of this sequence by user.
        block_size: Size of the cache block
        data: An SequenceData instance which holds the data.
    """
    
    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int
    ) -> None:
        """Initialization

        Args:
            seq_id (int): Unique id of this sequence.
            prompt (str): Prompt input by user.
            prompt_token_ids (List[int]): Token ids of the prompt.
            block_size (int): Size of cache block
        """        
        # TODO(MX) Maybe uuid should be better
        
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size
        
        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: List[Dict[int, float]] = []
        self.output_text = ""
        
        self.logical_token_blocks: List[LogicalTokenBlock] = []
        self._append_token_ids_to_blocks(prompt_token_ids)
        
        # Initial status
        self.status = SequenceStatus.WAITING
        
        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens in str form
        self.tokens: Optional[List[str]] = None
    
    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)
    
    def _append_token_ids_to_blocks(self, token_ids: List[int]) -> None:
        """Append tokens to logical blocks.
        
        New blocks will be allocated if no enough space is available.
        This method appends IDs to only block, not SequenceData
        """
        if not self.logical_token_blocks:
            self._append_logical_block()
        
        cursor = 0
        while cursor < len(token_ids):
            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]
            
            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor : cursor + num_empty_slots])
            
            cursor += num_empty_slots
    
    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        """Append 1 token id to the block list and SequenceData.
        
        This is the main API to add newly generated token to the Sequence.

        Args:
            token_id (int): Token id to be appended.
            logprobs (Dict[int, float]): The final log probability of the whole vocabulary.
        """
        assert token_id in logprobs
        self._append_token_ids_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        
        self.data.append_token_id(token_id, logprobs[token_id])
        
    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob
    
    def get_beam_search_score(
        self,
        length_penalty: float = 0.0,
        seq_len: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from
        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        
        Args:
            length
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if eos_token_id is not None and self.get_last_token_id() == eos_token_id:
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)
    
    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)
    
    def fork(self, new_seq_id: int) -> "Sequence":
        """Deep copy of the current Sequence.

        Args:
            new_seq_id (int): New Sequence id.

        Returns:
            Sequence: A new independent copy of the current Sequence.
        """
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq
    
    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")
            
    

class SequenceGroup:
    """A group of sequences that are generated from the same prompt.
    
    Attributes:
        request_id: ID of the request this SequenceGroup belongs to.
        seqs_dict: Dict[sequence_id, Sequence].
        sampling_params: Sampling parameters.
        arrival_time: Arrival time of the request this Sequence belongs to.
        prompt_logprobs: Log probabilities of the prompt.
    """
    
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.prompt_logprobs: Optional[List[Optional[Dict[int, float]]]] = None
    
    @property
    def prompt(self) -> str:
        # All sequences should possess the same prompt.
        return next(iter(self.seqs_dict.values())).prompt
    
    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should possess the same prompt.
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids
    
    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        """Get all sequences in designated status within this group.

        Args:
            status (Optional[SequenceStatus], optional): Designated status. 
                If is None, all sequences are returned.
                Defaults to None.

        Returns:
            List[Sequence]: All eligible sequences.
        """
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [seq for seq in self.seqs_dict.values() if seq.status == status]
        
    def get_unfinished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if not seq.is_finished()]
    
    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]
    
    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()
        
    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())
    
    def is_finished(self) -> bool:
        """Whether **ALL** sequences are finished."""
        return all(seq.is_finished() for seq in self.get_seqs())
    
    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]
    
    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")
        
class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. Dict(Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. Dict(Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables