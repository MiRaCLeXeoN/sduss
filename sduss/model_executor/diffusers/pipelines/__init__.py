from .stable_diffusion import StableDiffusionPipeline
from .stable_diffusion_xl import StableDiffusionXLPipeline

# Target_name -> (folder, class_name)
PipelneRegistry = {
    "StableDiffusionPipeline" : ("stable_diffusion", "StableDiffusionPipeline"),
    "StableDiffusionXLPipeline" : ("stable_diffusion_xl", "StableDiffusionXLPipeline"),
}