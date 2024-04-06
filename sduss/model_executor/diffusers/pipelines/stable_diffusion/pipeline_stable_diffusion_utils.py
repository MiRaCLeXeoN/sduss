from dataclasses import dataclass, fields, field
from typing import Union, Optional, List, Dict, Callable, Any, Type

import PIL
import numpy as np
import torch

from ..pipeline_utils import (BasePipelineStepInput, BasePipelinePostInput)

from sduss.model_executor.utils import BaseOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers.image_processor import PipelineImageInput
from sduss.logger import init_logger
from sduss.worker import WorkerRequest

logger = init_logger(__name__)


class StableDiffusionPipelinePrepareInput:
    @staticmethod
    def prepare_prepare_input(
        worker_reqs: List[WorkerRequest],
    ) -> Dict:
        input_dict: Dict = {}
        input_dict["worker_reqs"] = worker_reqs
        # These variables must be none right now
        input_dict["prompt"] = None
        input_dict["negative_prompt"] = None
        input_dict["num_inference_steps"] = None
        input_dict["latents"] = None
        input_dict["prompt_embeds"] = None
        input_dict["negative_prompt_embeds"] = None

        # Get a example sampling params
        sp: "StableDiffusionPipelineSamplingParams" = worker_reqs[0].sampling_params
        # These variables are all set as default, we can use them across all reqs.
        input_dict["height"] = sp.height  
        input_dict["width"] = sp.width  
        input_dict["timesteps"] = sp.timesteps  
        input_dict["guidance_scale"] = sp.guidance_scale  
        input_dict["eta"] = sp.eta  
        input_dict["generator"] = sp.generator  
        input_dict["ip_adapter_image"] = sp.ip_adapter_image  
        input_dict["ip_adapter_image_embeds"] = sp.ip_adapter_image_embeds  
        input_dict["output_type"] = sp.output_type  
        input_dict["return_dict"] = sp.return_dict  
        input_dict["cross_attention_kwargs"] = sp.cross_attention_kwargs  
        input_dict["guidance_rescale"] = sp.guidance_rescale  
        input_dict["clip_skip"] = sp.clip_skip  
        input_dict["callback_on_step_end"] = sp.callback_on_step_end  
        input_dict["callback_on_step_end_tensor_inputs"] = sp.callback_on_step_end_tensor_inputs  
    
        return input_dict
        

@dataclass
class StableDiffusionPipelinePrepareOutput:
    """Params that are same as sampling_params will not be stored here."""
    timestep_cond: torch.Tensor
    added_cond_kwargs: Optional[Dict]
    extra_step_kwargs: Dict
    device: torch.device
    do_classifier_free_guidance: bool
    # latents: torch.Tensor  # update to sampling_params
    # prompt_embeds: torch.FloatTensor  # update to sampling_params


class StableDiffusionPipelineStepInput(BasePipelineStepInput):
    @staticmethod
    def prepare_step_input(
        worker_reqs: List[WorkerRequest],
    ) -> Dict: 
        """Prepare input parameters for denoising_step function.

        Args:
            worker_reqs (List[WorkerRequest]): Reqs to be batched.

        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}
        input_dict["worker_reqs"] = worker_reqs
        
        sp: "StableDiffusionPipelineSamplingParams" = worker_reqs[0].sampling_params
        # params from sampling_params
        input_dict["callback_on_step_end"] = sp.callback_on_step_end
        input_dict["callback_on_step_end_tensor_inputs"] = sp.callback_on_step_end_tensor_inputs
        input_dict["guidance_rescale"] = sp.guidance_scale
        input_dict["guidance_scale"] = sp.guidance_rescale
        input_dict["cross_attention_kwargs"] = sp.cross_attention_kwargs

        po: "StableDiffusionPipelinePrepareOutput" = worker_reqs[0].prepare_output
        # params from prepare_output
        input_dict["timestep_cond"] = po.timestep_cond
        input_dict["added_cond_kwargs"] = po.added_cond_kwargs
        input_dict["extra_step_kwargs"] = po.extra_step_kwargs
        input_dict["do_classifier_free_guidance"] = po.do_classifier_free_guidance
    
        return input_dict
        

@dataclass
class StableDiffusionPipelineStepOutput:
    """Step output class.
    
    For this pipeline, nothing should be stored.
    """
    pass


class StableDiffusionPipelinePostInput(BasePipelinePostInput):
    @staticmethod
    def prepare_post_input(
        worker_reqs: List[WorkerRequest],
    ) -> Dict:
        """Prepare input parameters for post_inference function.

        Args:
            worker_reqs (List[WorkerRequest]): Reqs to be batched.

        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}
        input_dict["worker_reqs"] = worker_reqs
        # params from sampling params
        sp: "StableDiffusionPipelineSamplingParams" = worker_reqs[0].sampling_params
        input_dict["output_type"] = sp.output_type
        input_dict["prompt_embeds_dtype"] = sp.prompt_embeds.dtype
        input_dict["generator"] = sp.generator
        # params from prepare output
        po: "StableDiffusionPipelinePrepareOutput" = worker_reqs[0].prepare_output
        input_dict["device"] = po.device

        return input_dict


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class StableDiffusionPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that vary
    # Defined in BaseSampling Params
    volatile_params = {
        "timesteps" : None,
        "guidance_scale" : 7.5,
        "eta" : 0.0,
        "generator" : None,
        "ip_adapter_image" : None,
        "ip_adapter_image_embeds" : None,
        "output_type" : "pil",
        "return_dict" : True,
        "cross_attention_kwargs" : None,
        "guidance_rescale" : 0.0,
        "clip_skip" : None,
        "callback_on_step_end" : None,
        "callback_on_step_end_tensor_inputs" : ["latents"]
    }
    
    utils_cls = {
        "prepare_input" : StableDiffusionPipelinePrepareInput,
        "prepare_output" : StableDiffusionPipelinePrepareOutput,
        "step_input" : StableDiffusionPipelineStepInput,
        "step_output" : StableDiffusionPipelineStepOutput,
        "post_input" : StableDiffusionPipelinePostInput,
        "pipeline_output" : StableDiffusionPipelineOutput,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height: Optional[int] = self.resolution
        self.width: Optional[int] = self.resolution

        self.timesteps: List[int] = kwargs.pop("timesteps", None)
        self.guidance_scale: float = kwargs.pop("guidance_scale", 7.5)
        self.eta: float = kwargs.pop("eta", 0.0)
        self.generator: Optional[Union[torch.Generator, List[torch.Generator]]] = kwargs.pop("generator", None)
        self.ip_adapter_image: Optional[PipelineImageInput] = kwargs.pop("ip_adapter_image", None)
        self.ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = kwargs.pop("ip_adapter_image_embeds", None)
        self.output_type: Optional[str] = kwargs.pop("output_type", "pil")
        self.return_dict: bool = kwargs.pop("return_dict", True)
        self.cross_attention_kwargs: Optional[Dict[str, Any]] = kwargs.pop("cross_attention_kwargs", None)
        self.guidance_rescale: float = kwargs.pop("guidance_rescale", 0.0)
        self.clip_skip: Optional[int] = kwargs.pop("clip_skip", None)
        self.callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = kwargs.pop("callback_on_step_end", None)
        self.callback_on_step_end_tensor_inputs: List[str] = kwargs.pop("callback_on_step_end_tensor_inputs", ["latents"])
        self._check_volatile_params()

    
    def _check_volatile_params(self):
        """Check volatile params to ensure they are the same as default."""
        for param_name in self.volatile_params:
            if getattr(self, param_name) != self.volatile_params[param_name]:
                raise RuntimeError(f"Currently, we do not support customized {param_name} parameter.")