from dataclasses import dataclass, fields, field
from typing import Union, Optional, List, Dict, Callable, Any, Type, Tuple, TYPE_CHECKING

import PIL
import numpy as np
import torch

from ..pipeline_utils import (BasePipelineStepInput, BasePipelinePostInput, BasePipelinePrepareOutput)

from sduss.model_executor.utils import BaseOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.logger import init_logger

from ...image_processor import PipelineImageInput

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sduss.worker.runner.wrappers import RunnerRequestDictType

class StableDiffusion3EsymredPipelinePrepareInput:
    @staticmethod
    def prepare_prepare_input(
        runner_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict:
        input_dict: Dict = {}
        input_dict["runner_reqs"] =  runner_reqs

        # Get a example sampling params
        sp: "StableDiffusion3EsymredPipelineSamplingParams" = runner_reqs[next(iter(runner_reqs.keys()))][0].sampling_params
        # These variables are all set as default, we can use them across all reqs.
        input_dict["guidance_scale"] = sp.guidance_scale
        input_dict["generator"] = sp.generator
        input_dict["pooled_prompt_embeds"] = sp.pooled_prompt_embeds
        input_dict["negative_pooled_prompt_embeds"] = sp.negative_pooled_prompt_embeds
        input_dict["joint_attention_kwargs"] = sp.joint_attention_kwargs
        input_dict["clip_skip"] = sp.clip_skip
        input_dict["max_sequence_length"] = sp.max_sequence_length
        input_dict["skip_layer_guidance_scale"] = sp.skip_layer_guidance_scale
        return input_dict
        

@dataclass
class StableDiffusion3EsymredPipelinePrepareOutput(BasePipelinePrepareOutput):
    """Params that are same as sampling_params will not be stored here."""
    pooled_prompt_embeds : torch.Tensor
    negative_pooled_prompt_embeds : torch.Tensor
    device: torch.device
    do_classifier_free_guidance: bool
    # latents: torch.Tensor  # update to sampling_params
    # prompt_embeds: torch.FloatTensor  # update to sampling_params

    def to_device(self, device) -> None:
        self.device = device
        self.pooled_prompt_embeds = self.pooled_prompt_embeds.to(device=device)
        self.negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.to(device=device)
        self.add_time_ids = self.add_time_ids.to(device=device)
        self.negative_add_time_ids = self.negative_add_time_ids.to(device=device)
    
    
    def to_dtype(self, dtype) -> None:
        self.pooled_prompt_embeds = self.pooled_prompt_embeds.to(dtype=dtype)
        self.negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.to(dtype=dtype)
        self.add_time_ids = self.add_time_ids.to(dtype=dtype)
        self.negative_add_time_ids = self.negative_add_time_ids.to(dtype=dtype)
    
    
    def to_numpy(self) -> None:
        self.pooled_prompt_embeds = self.pooled_prompt_embeds.numpy()
        self.negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.numpy()
        self.add_time_ids = self.add_time_ids.numpy()
        self.negative_add_time_ids = self.negative_add_time_ids.numpy()
    
    
    def to_tensor(self) -> None:
        self.pooled_prompt_embeds = torch.from_numpy(self.pooled_prompt_embeds)
        self.negative_pooled_prompt_embeds = torch.from_numpy(self.negative_pooled_prompt_embeds)
        self.add_time_ids = torch.from_numpy(self.add_time_ids)
        self.negative_add_time_ids = torch.from_numpy(self.negative_add_time_ids)


class StableDiffusion3EsymredPipelineStepInput(BasePipelineStepInput):
    @staticmethod
    def prepare_step_input(
        runner_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict: 
        """Prepare input parameters for denoising_step function.

        Args:
            runner_reqs (List[RunnerRequestDictType]): Reqs to be batched.

        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}

        runner_reqs_dict = {}
        for res in runner_reqs:
            runner_reqs_dict[str(res)] = runner_reqs[res]
        input_dict["runner_reqs"] = runner_reqs_dict
        
        # params from sampling_params
        sp: "StableDiffusion3EsymredPipelineSamplingParams" = runner_reqs[res][0].sampling_params
        input_dict["guidance_scale"] = sp.guidance_scale

        # params from prepare output
        po: "StableDiffusion3EsymredPipelinePrepareOutput" = runner_reqs[res][0].prepare_output
        input_dict["do_classifier_free_guidance"] = po.do_classifier_free_guidance

        input_dict["is_sliced"] = kwargs.pop("is_sliced")
        input_dict["patch_size"] = kwargs.pop("patch_size")
    
        return input_dict
        

@dataclass
class StableDiffusion3EsymredPipelineStepOutput:
    """Step output class.
    
    For this pipeline, nothing should be stored.
    """
    pass


class StableDiffusion3EsymredPipelinePostInput(BasePipelinePostInput):
    @staticmethod
    def prepare_post_input(
        runner_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict:
        """Prepare input parameters for post_inference function.

        Args:
            runner_reqs (List[RunnerRequestDictType]): Reqs to be batched.
        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}

        input_dict["runner_reqs"] = runner_reqs

        sp: "StableDiffusion3EsymredPipelineSamplingParams" = runner_reqs[next(iter(runner_reqs.keys()))][0].sampling_params
        input_dict["output_type"] = sp.output_type

        return input_dict


@dataclass
class StableDiffusion3EsymredPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]


class StableDiffusion3EsymredPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that vary
    # Defined in BaseSampling Params
    volatile_params = {
        "sigmas": None,
        "guidance_scale": 7.0,
        "generator": None,
        "pooled_prompt_embeds": None,
        "negative_pooled_prompt_embeds": None,
        "ip_adapter_image": None,
        "ip_adapter_image_embeds": None,
        "output_type": "pil",
        "return_dict": True,
        "joint_attention_kwargs": None,
        "clip_skip": None,
        "callback_on_step_end": None,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "max_sequence_length": 256,
        "skip_guidance_layers": None,
        "skip_layer_guidance_scale": 2.8,
        "skip_layer_guidance_stop": 0.2,
        "skip_layer_guidance_start": 0.01,
        "mu": None,
    }
    
    utils_cls = {
        "prepare_input" : StableDiffusion3EsymredPipelinePrepareInput,
        "prepare_output" : StableDiffusion3EsymredPipelinePrepareOutput,
        "step_input" : StableDiffusion3EsymredPipelineStepInput,
        "step_output" : StableDiffusion3EsymredPipelineStepOutput,
        "post_input" : StableDiffusion3EsymredPipelinePostInput,
        "pipeline_output" : StableDiffusion3EsymredPipelineOutput,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height: Optional[int] = self.resolution
        self.width: Optional[int] = self.resolution

        # Specific params that can be mutable
        self.prompt_2: Optional[Union[str, List[str]]] = kwargs.pop("prompt_2", self.prompt)
        self.prompt_3: Optional[Union[str, List[str]]] = kwargs.pop("prompt_3", self.prompt)
        self.negative_prompt_2: Optional[Union[str, List[str]]] = kwargs.pop("negative_prompt_2", self.negative_prompt)
        self.negative_prompt_3: Optional[Union[str, List[str]]] = kwargs.pop("negative_prompt_3", self.negative_prompt)

        # Volatiles ones
        self.sigmas: Optional[List[float]] = self._get_volatile_params_from_kwargs("sigmas", kwargs)
        self.guidance_scale: float = self._get_volatile_params_from_kwargs("guidance_scale", kwargs)
        self.generator: Optional[Union[torch.Generator, List[torch.Generator]]] = self._get_volatile_params_from_kwargs("generator", kwargs)
        self.pooled_prompt_embeds: Optional[torch.FloatTensor] = self._get_volatile_params_from_kwargs("pooled_prompt_embeds", kwargs)
        self.negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = self._get_volatile_params_from_kwargs("negative_pooled_prompt_embeds", kwargs)
        self.ip_adapter_image: Optional[PipelineImageInput] = self._get_volatile_params_from_kwargs("ip_adapter_image", kwargs)
        self.ip_adapter_image_embeds: Optional[torch.Tensor] = self._get_volatile_params_from_kwargs("ip_adapter_image_embeds", kwargs)
        self.output_type: Optional[str] = self._get_volatile_params_from_kwargs("output_type", kwargs)
        self.return_dict: bool = self._get_volatile_params_from_kwargs("return_dict", kwargs)
        self.joint_attention_kwargs: Optional[Dict[str, Any]] = self._get_volatile_params_from_kwargs("joint_attention_kwargs", kwargs)
        self.clip_skip: Optional[int] = self._get_volatile_params_from_kwargs("clip_skip", kwargs)
        self.callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = self._get_volatile_params_from_kwargs("callback_on_step_end", kwargs)
        self.callback_on_step_end_tensor_inputs: List[str] = self._get_volatile_params_from_kwargs("callback_on_step_end_tensor_inputs", kwargs)
        self.max_sequence_length: int = self._get_volatile_params_from_kwargs("max_sequence_length", kwargs)
        self.skip_guidance_layers: List[int] = self._get_volatile_params_from_kwargs("skip_guidance_layers", kwargs)
        self.skip_layer_guidance_scale: float = self._get_volatile_params_from_kwargs("skip_layer_guidance_scale", kwargs)
        self.skip_layer_guidance_stop: float = self._get_volatile_params_from_kwargs("skip_layer_guidance_stop", kwargs)
        self.skip_layer_guidance_start: float = self._get_volatile_params_from_kwargs("skip_layer_guidance_start", kwargs)
        self.mu: Optional[float] = self._get_volatile_params_from_kwargs("mu", kwargs)

        self._check_volatile_params()


    def is_compatible_with(self, sampling_params: "StableDiffusion3EsymredPipelineSamplingParams") -> bool:
        is_compatible = True
        # Only volatile params are sure to influce the compatibility
        # But since we've fixed them, params must be compatible
        return is_compatible and super().is_compatible_with(sampling_params)

    
    def _check_volatile_params(self):
        """Check volatile params to ensure they are the same as default."""
        for param_name in self.volatile_params:
            if getattr(self, param_name) != self.volatile_params[param_name]:
                raise RuntimeError(f"Currently, we do not support customized {param_name} parameter.")
    
    
    def to_device(self, device) -> None:
        return super().to_device(device)
    
    
    def to_dtype(self, dtype) -> None:
        return super().to_dtype(dtype)
    
    
    def to_tensor(self) -> None:
        return super().to_tensor()
    
    
    def to_numpy(self) -> None:
        return super().to_numpy()