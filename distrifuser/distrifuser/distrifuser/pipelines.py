import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusion3Pipeline
from diffusers.models import SD3Transformer2DModel
import inspect
from typing import Optional, List, Tuple, Union, Dict, Any, Callable
from .models.distri_sdxl_unet_pp import DistriUNetPP
from .models.distri_sdxl_unet_tp import DistriUNetTP
from .models.naive_patch_sdxl import NaivePatchUNet
from .models.distri_sd3_transformer_pp import DistriTransformerPP
from .utils import DistriConfig, PatchParallelismCommManager

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class DistriSD3Pipeline:
    def __init__(self, pipeline: StableDiffusion3Pipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        # self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-3.5-medium"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="transformer"
        ).to(device)

        if distri_config.parallelism == "patch":
            transformer = DistriTransformerPP(transformer, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, transformer=transformer, **kwargs
        ).to(device)
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        return DistriSD3Pipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, height, width, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.prepare(height, width, batch_size=len(kwargs.get("prompt")))
        self.pipeline.transformer.set_counter(0)
        return self.pipeline(height=height, width=width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, height, width, batch_size: int = 1, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        # height = distri_config.height
        # width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        # prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        #     prompt="",
        #     prompt_2=None,
        #     device=device,
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=False,
        #     negative_prompt=None,
        #     negative_prompt_2=None,
        #     prompt_embeds=None,
        #     negative_prompt_embeds=None,
        #     pooled_prompt_embeds=None,
        #     negative_pooled_prompt_embeds=None,
        # )
        single_batch_size = batch_size
        batch_size = batch_size * 2 if distri_config.do_classifier_free_guidance else batch_size
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=[""] * single_batch_size,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self.distri_config.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
        )

        num_channels_latents = self.pipeline.transformer.config.in_channels
        # latents = pipeline.prepare_latents(
        #     batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        # )
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            None,
        )
        scheduler_kwargs = {}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            device=device,
            sigmas=None,
            num_inference_steps=50,
            **scheduler_kwargs,
        )
        if self.distri_config.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)
        self.pipeline._num_timesteps = len(timesteps)

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["hidden_states"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        static_inputs["pooled_projections"] = pooled_prompt_embeds
        static_inputs["joint_attention_kwargs"] = None
        # static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.transformer.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.transformer.set_counter(0)
            pipeline.transformer(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.transformer.set_counter(0)
        pipeline.transformer(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.transformer.set_counter(counter)
                    output = pipeline.transformer(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                torch.cuda.synchronize()
                cuda_graphs.append(graph)
            pipeline.transformer.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs


class DistriSDXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        # self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        return DistriSDXLPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, height, width, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.prepare(height, width, batch_size=len(kwargs.get("prompt")))
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=height, width=width, *args, **kwargs)
    
    @torch.no_grad()
    def prepare(self, height, width, batch_size: int = 1, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        # height = distri_config.height
        # width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
        # TODO(MX): modifiable batch size
        batch_size = batch_size * 2 if distri_config.do_classifier_free_guidance else batch_size

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1, 1)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            add_text_embeds = add_text_embeds.repeat(batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            # This only sets
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            # ! Dummy run
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                torch.cuda.synchronize()
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs


class DistriSDPipeline:
    def __init__(self, pipeline: StableDiffusionPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        return DistriSDPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if not "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        device = distri_config.device

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            "",
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=kwargs.get("clip_skip", None),
        )

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        prompt_embeds = prompt_embeds.to(device)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs
