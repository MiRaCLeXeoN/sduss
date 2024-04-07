from sduss import DiffusionPipeline

# 1. Create pipeline
pipe = DiffusionPipeline()

sampling_params_cls = pipe.get_sampling_params_cls()

sampling_params = []

sampling_params.append(sampling_params_cls("astrount riding a horse on the moon."))
sampling_params.append(sampling_params_cls("a flowring sitting on the crest."))

outputs = pipe.generate(sampling_params)
