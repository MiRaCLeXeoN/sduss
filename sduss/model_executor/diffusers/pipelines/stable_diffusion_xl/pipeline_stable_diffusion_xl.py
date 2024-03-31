from typing import Dict

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import(
    StableDiffusionXLPipeline as DiffusersStableDiffusionXLPipeline)

from ..pipeline_utils import BasePipeline

class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline, BasePipeline):

    @classmethod
    def instantiate_pipeline(cls, **kwargs):
        sub_modules: Dict = kwargs.pop("sub_modules", {})
        return cls(**sub_modules)
