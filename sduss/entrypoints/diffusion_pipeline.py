from typing import Optional, Union, List, Tuple

import tqdm

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sduss.utils import Counter
from sduss.entrypoints.outputs import RequestOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers import BasePipeline
from sduss.engine.arg_utils import EngineArgs
from sduss.engine.engine import Engine

class DiffusionPipeline:

    def __init__(
        self,
        model_name_or_pth: str,
        **kwargs,
    ) -> None:
        if "disable_log_status" not in kwargs:
            kwargs["disable_log_status"] = True
        engine_args = EngineArgs(model_name_or_pth, **kwargs)
        self.engine = Engine.from_engine_args(engine_args)
        self.request_counter = Counter()

        # Set afterwards
        self.pipeline_cls = None
        self.sampling_param_cls = None
    
    def get_sampling_params_cls(self):
        if self.sampling_param_cls is not None:
            return self.sampling_param_cls

        from sduss.model_executor.model_loader import get_pipeline_cls
        self.pipeline_cls: BasePipeline = get_pipeline_cls(self.engine.pipeline_config)
        self.sampling_param_cls = self.pipeline_cls.get_sampling_params_cls()
        return self.sampling_param_cls
        
    def generate(
        self,
        sampling_params: Union[BaseSamplingParams, List[BaseSamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates images according to prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            sampling_params: The sampling parameters for diffusion procedure. Each prompt can have
                its own sampling parameter. The provided number must be either 1 or len(prompts).
                If 1 is provided, the sampling params will be applied to all prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            images in the same order as the input prompts.
        """
        if sampling_params is None:
            raise ValueError("Sampling parameters must be provided.")

        # Add requests to the engine
        num_requests = len(sampling_params)
        for i in range(num_requests):
            assert isinstance(sampling_params[i], self.sampling_param_cls)
            self._add_request_to_engine(sampling_params[i])
        
        return self._run_engine(use_tqdm)  

    
    def _add_request_to_engine(
        self,
        sampling_params: BaseSamplingParams,
    ) -> None:
        request_id = next(self.request_counter)
        self.engine.add_request(
            request_id,
            sampling_params)


    def _run_engine(
        self,
        use_tqdm: bool,
    ) -> List[RequestOutput]:
        if use_tqdm:
            num_requests = self.engine.get_num_unfinished_requests()
            pbar = tqdm.tqdm(total=num_requests, desc="Processed requests")       
        
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