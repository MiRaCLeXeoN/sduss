from typing import Optional, Union, List, Tuple

import tqdm

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sduss.utils import Counter
from sduss.outputs import RequestOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.engine.arg_utils import EngineArgs
from sduss.engine.engine import Engine

class DiffusionPipeline:

    def __init__(
        self,
        model_name_or_pth: str,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(model_name_or_pth, **kwargs)
        self.engine = Engine.from_engine_args(engine_args)
        self.request_counter = Counter()
        
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[BaseSamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates images according to prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts based on which sampling will be performed.
            negative_prompts: A list of negative prompts based on which sampling will be performed.
            sampling_params: The sampling parameters for diffusion procedure. 
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            images in the same order as the input prompts.
        """
        if prompts is None:
            raise ValueError("Prompts must be provided.")
        if sampling_params is None:
            raise ValueError("Sampling parameters must be provided.")
        if len(prompts) != len(negative_prompts):
            raise ValueError(f"The length of prompts(len={len(prompts)}) doesn't equal to "
                             f"the length of negative prompts(len={len(negative_prompts)}).")
        

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Add requests to the engine
        num_requests = len(prompts)
        for i in range(num_requests):
            self._add_request_to_engine(prompts[i], negative_prompts[i], sampling_params)
        
        return self._run_engine(use_tqdm)  
    
    def _add_request_to_engine(
        self,
        prompt: Optional[str],
        negative_prompt: Optional[str],
        sampling_params: BaseSamplingParams,
    ) -> None:
        request_id = next(self.request_counter)
        self.engine.add_request(
            request_id,
            prompt,
            negative_prompt,
            sampling_params)

    def _run_engine(
        self,
        use_tqdm: bool,
    ) -> List[RequestOutput]:
        if use_tqdm:
            num_requests = self.engine.get_num_unfinished_requests()
            pbar = tqdm.tqdm(total=num_requests, desc="Processed prompts")       
        
        outputs: List[RequestOutput] = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
            
        # Sort the requests by request ID
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
        
