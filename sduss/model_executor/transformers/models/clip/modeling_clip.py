from transformers import (
    CLIPTextModelWithProjection as TransformersCLIPTextModelWithProjection,
    CLIPTextModel as TransformersCLIPTextModel,
)


class CLIPTextModel(TransformersCLIPTextModel):
    pass


class CLIPTextModelWithProjection(TransformersCLIPTextModelWithProjection):
    pass