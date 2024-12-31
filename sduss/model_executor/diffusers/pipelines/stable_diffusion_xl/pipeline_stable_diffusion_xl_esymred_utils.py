from dataclasses import dataclass, fields, field
from typing import Union, Optional, List, Dict, Callable, Any, Type, Tuple, TYPE_CHECKING

import PIL
import numpy as np
import torch

from ..pipeline_utils import (BasePipelineStepInput, BasePipelinePostInput, BasePipelinePrepareOutput)

from sduss.model_executor.utils import BaseOutput
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers.image_processor import PipelineImageInput
from sduss.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sduss.worker.runner.wrappers import RunnerRequestDictType

class StableDiffusionXLEsymredPipelinePrepareInput:
    @staticmethod
    def prepare_prepare_input(
        worker_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict:
        input_dict: Dict = {}
        input_dict["worker_reqs"] =  worker_reqs

        # Get a example sampling params
        sp: "StableDiffusionXLEsymredPipelineSamplingParams" = worker_reqs[next(iter(worker_reqs.keys()))][0].sampling_params
        # These variables are all set as default, we can use them across all reqs.
        input_dict["denoising_end"] = sp.denoising_end
        input_dict["guidance_scale"] = sp.guidance_scale
        input_dict["eta"] = sp.eta
        input_dict["generator"] = sp.generator
        input_dict["pooled_prompt_embeds"] = sp.pooled_prompt_embeds
        input_dict["negative_pooled_prompt_embeds"] = sp.negative_pooled_prompt_embeds
        input_dict["ip_adapter_image"] = sp.ip_adapter_image
        input_dict["ip_adapter_image_embeds"] = sp.ip_adapter_image_embeds
        input_dict["output_type"] = sp.output_type
        input_dict["return_dict"] = sp.return_dict
        input_dict["cross_attention_kwargs"] = sp.cross_attention_kwargs
        input_dict["guidance_rescale"] = sp.guidance_rescale
        input_dict["crops_coords_top_left"] = sp.crops_coords_top_left
        input_dict["negative_original_size"] = sp.negative_original_size
        input_dict["negative_crops_coords_top_left"] = sp.negative_crops_coords_top_left
        input_dict["negative_target_size"] = sp.negative_target_size
        input_dict["clip_skip"] = sp.clip_skip
    
        return input_dict
        

@dataclass
class StableDiffusionXLEsymredPipelinePrepareOutput(BasePipelinePrepareOutput):
    """Params that are same as sampling_params will not be stored here."""
    pooled_prompt_embeds : torch.Tensor
    negative_pooled_prompt_embeds : torch.Tensor
    add_time_ids : torch.Tensor
    negative_add_time_ids : torch.Tensor
    timestep_cond: torch.Tensor
    extra_step_kwargs: Dict
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


class StableDiffusionXLEsymredPipelineStepInput(BasePipelineStepInput):
    @staticmethod
    def prepare_step_input(
        worker_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict: 
        """Prepare input parameters for denoising_step function.

        Args:
            worker_reqs (List[RunnerRequestDictType]): Reqs to be batched.

        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}

        worker_reqs_dict = {}
        for res in worker_reqs:
            worker_reqs_dict[str(res)] = worker_reqs[res]
        input_dict["worker_reqs"] = worker_reqs_dict
        
        # params from sampling_params
        sp: "StableDiffusionXLEsymredPipelineSamplingParams" = worker_reqs[res][0].sampling_params
        input_dict["guidance_rescale"] = sp.guidance_rescale
        input_dict["guidance_scale"] = sp.guidance_scale
        input_dict["cross_attention_kwargs"] = sp.cross_attention_kwargs
        input_dict["ip_adapter_image"] = sp.ip_adapter_image
        input_dict["ip_adapter_image_embeds"] = sp.ip_adapter_image_embeds

        # params from prepare output
        po: "StableDiffusionXLEsymredPipelinePrepareOutput" = worker_reqs[res][0].prepare_output
        input_dict["do_classifier_free_guidance"] = po.do_classifier_free_guidance
        input_dict["timestep_cond"] = po.timestep_cond
        input_dict["extra_step_kwargs"] = po.extra_step_kwargs

        input_dict["is_sliced"] = kwargs.pop("is_sliced")
        input_dict["patch_size"] = kwargs.pop("patch_size")
    
        return input_dict
        

@dataclass
class StableDiffusionXLEsymredPipelineStepOutput:
    """Step output class.
    
    For this pipeline, nothing should be stored.
    """
    pass


class StableDiffusionXLEsymredPipelinePostInput(BasePipelinePostInput):
    @staticmethod
    def prepare_post_input(
        worker_reqs: "RunnerRequestDictType",
        **kwargs,
    ) -> Dict:
        """Prepare input parameters for post_inference function.

        Args:
            worker_reqs (List[RunnerRequestDictType]): Reqs to be batched.
        Returns:
            Dict: kwargs dict as input.
        """
        input_dict: Dict = {}

        input_dict["worker_reqs"] = worker_reqs

        sp: "StableDiffusionXLEsymredPipelineSamplingParams" = worker_reqs[next(iter(worker_reqs.keys()))][0].sampling_params
        input_dict["output_type"] = sp.output_type

        return input_dict


@dataclass
class StableDiffusionXLEsymredPipelineOutput(BaseOutput):
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


class StableDiffusionXLEsymredPipelineSamplingParams(BaseSamplingParams):
    """Sampling parameters for StableDiffusionPipeline."""
    # Params that vary
    # Defined in BaseSampling Params
    volatile_params = {
        "pooled_prompt_embeds" : None,
        "negative_pooled_prompt_embeds" : None,
        "timesteps" : None,
        "denoising_end" : None,
        "guidance_scale" : 5.0,
        "eta" : 0.0,
        "generator" : None,
        "ip_adapter_image" : None,
        "ip_adapter_image_embeds" : None,
        "output_type" : "pil",
        "return_dict" : True,
        "cross_attention_kwargs" : None,
        "guidance_rescale" : 0.0,
        "original_size" : None,
        "crops_coords_top_left" : (0, 0),
        "target_size" : None,
        "negative_original_size" : None,
        "negative_crops_coords_top_left" : (0, 0),
        "negative_target_size" : None,
        "clip_skip" : None,
        "callback_on_step_end" : None,
        "callback_on_step_end_tensor_inputs" : ["latents"],
    }
    
    utils_cls = {
        "prepare_input" : StableDiffusionXLEsymredPipelinePrepareInput,
        "prepare_output" : StableDiffusionXLEsymredPipelinePrepareOutput,
        "step_input" : StableDiffusionXLEsymredPipelineStepInput,
        "step_output" : StableDiffusionXLEsymredPipelineStepOutput,
        "post_input" : StableDiffusionXLEsymredPipelinePostInput,
        "pipeline_output" : StableDiffusionXLEsymredPipelineOutput,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height: Optional[int] = self.resolution
        self.width: Optional[int] = self.resolution

        # Specific params that can be mutable
        self.prompt_2: Optional[Union[str, List[str]]] = kwargs.pop("prompt_2", "")
        self.negative_prompt_2: Optional[Union[str, List[str]]] = kwargs.pop("negative_prompt_2", "")

        self.original_size: Optional[Tuple[int, int]] = self._get_volatile_params_from_kwargs("original_size", kwargs)
        self.pooled_prompt_embeds: Optional[torch.FloatTensor] = self._get_volatile_params_from_kwargs("pooled_prompt_embeds", kwargs)
        self.negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = self._get_volatile_params_from_kwargs("negative_pooled_prompt_embeds", kwargs)
        self.timesteps: List[int] = self._get_volatile_params_from_kwargs("timesteps", kwargs)
        self.denoising_end: Optional[float] = self._get_volatile_params_from_kwargs("denoising_end", kwargs)
        self.guidance_scale: float = self._get_volatile_params_from_kwargs("guidance_scale", kwargs)
        self.eta: float = self._get_volatile_params_from_kwargs("eta", kwargs)
        self.generator: Optional[Union[torch.Generator, List[torch.Generator]]] = self._get_volatile_params_from_kwargs("generator", kwargs)
        self.ip_adapter_image: Optional[PipelineImageInput] = self._get_volatile_params_from_kwargs("ip_adapter_image", kwargs)
        self.ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = self._get_volatile_params_from_kwargs("ip_adapter_image_embeds", kwargs)
        self.output_type: Optional[str] = self._get_volatile_params_from_kwargs("output_type", kwargs)
        self.return_dict: bool = self._get_volatile_params_from_kwargs("return_dict", kwargs)
        self.cross_attention_kwargs: Optional[Dict[str, Any]] = self._get_volatile_params_from_kwargs("cross_attention_kwargs", kwargs)
        self.guidance_rescale: float = self._get_volatile_params_from_kwargs("guidance_rescale", kwargs)
        self.crops_coords_top_left: Tuple[int, int] = self._get_volatile_params_from_kwargs("crops_coords_top_left", kwargs)
        self.target_size: Optional[Tuple[int, int]] = self._get_volatile_params_from_kwargs("target_size", kwargs)
        self.negative_original_size: Optional[Tuple[int, int]] = self._get_volatile_params_from_kwargs("negative_original_size", kwargs)
        self.negative_crops_coords_top_left: Tuple[int, int] = self._get_volatile_params_from_kwargs("negative_crops_coords_top_left", kwargs)
        self.negative_target_size: Optional[Tuple[int, int]] = self._get_volatile_params_from_kwargs("negative_target_size", kwargs)
        self.clip_skip: Optional[int] = self._get_volatile_params_from_kwargs("clip_skip", kwargs)
        self.callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = self._get_volatile_params_from_kwargs("callback_on_step_end", kwargs)
        self.callback_on_step_end_tensor_inputs: List[str] = self._get_volatile_params_from_kwargs("callback_on_step_end_tensor_inputs", kwargs)

        self._check_volatile_params()


    def is_compatible_with(self, sampling_params: "StableDiffusionXLEsymredPipelineSamplingParams") -> bool:
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