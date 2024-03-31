# Target_name -> (folder, class_name)
from .pipeline_utils import BasePipeline

from .stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionSafetyChecker,
    ESyMReDStableDiffusionPipeline,
)

from .stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    ESyMReDStableDiffusionXLPipeline,
)



PipelneRegistry = {
    "StableDiffusionPipeline" : ("stable_diffusion", "StableDiffusionPipeline"),
    "StableDiffusionXLPipeline" : ("stable_diffusion_xl", "StableDiffusionXLPipeline"),
}

EsyMReDPipelineRegistry = {
    "StableDiffusionPipeline" : ("stable_diffusion", "ESyMReDStableDiffusionPipeline"),
    "StableDiffusionXLPipeline" : ("stable_diffusion_xl", "ESyMReDStableDiffusionXLPipeline"),
}