import json
import os

POSTPROCESSING_DDL = None
DENOISING_DDL = None
DISCARD_SLACK = None
STANDALONE = None
Hyper_Parameter = None

path = __file__
path = os.path.dirname(path)
path += "/configs/esymred.json"

with open(path, "r") as f:
    data = json.load(f)
    # Read
    Hyper_Parameter = data["Hyper_Parameter"]
    STANDALONE = data["STANDALONE"]
    DISCARD_SLACK = data["DISCARD_SLACK"]

    # Envs
    SLO = os.getenv("SLO")
    assert SLO is not None
    SLO = int(SLO)

    # Calculate ddl
    POSTPROCESSING_DDL = { }
    for model_name in STANDALONE:
        POSTPROCESSING_DDL[model_name] = { }
        resolutions = list(STANDALONE[model_name]["denoising"].keys())
        for res in resolutions:
            POSTPROCESSING_DDL[model_name][res] = (
                (STANDALONE[model_name]["denoising"][res] +
                STANDALONE[model_name]["postprocessing"][res]) * SLO
            )

    DENOISING_DDL = {}
    for model_name in STANDALONE:
        DENOISING_DDL[model_name] = { }
        resolutions = list(STANDALONE[model_name]["denoising"].keys())
        for res in resolutions:
            DENOISING_DDL[model_name][res] = (
                (STANDALONE[model_name]["denoising"][res]) * SLO
            )