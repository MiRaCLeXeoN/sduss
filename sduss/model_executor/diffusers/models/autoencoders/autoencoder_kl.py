from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from torch import FloatTensor, Generator
from torch._C import FloatTensor, Generator

class AutoencoderKL(DiffusersAutoencoderKL):
    
    def forward(
        self, 
        sample: FloatTensor, 
        sample_posterior: bool = False, 
        return_dict: bool = True, 
        generator: Generator | None = None
    ) -> DecoderOutput | FloatTensor:
        return super().forward(sample, sample_posterior, return_dict, generator)

