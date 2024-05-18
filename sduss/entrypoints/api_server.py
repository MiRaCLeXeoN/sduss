import argparse
import json
import sys
from typing import AsyncGenerator, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, FileResponse
import uvicorn

from sduss.engine.arg_utils import AsyncEngineArgs
from sduss.engine.async_engine import AsyncEngine
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.model_executor.diffusers import BasePipeline
from sduss.utils import random_uuid
from sduss.model_executor.model_loader import get_pipeline_cls
from sduss.entrypoints.outputs import RequestOutput

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
pipeline_cls = None
sampling_param_cls = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict: Dict = await request.json()

    request_id = random_uuid()
    sampling_params = sampling_param_cls(**request_dict)

    results_generator = engine.generate(request_id=request_id, sampling_params=sampling_params)

    # Non-streaming case. Iterate only once.
    final_output: RequestOutput = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort_req(request_id)
            return Response(status_code=499)
        final_output = request_output
    assert final_output is not None
    
    if not final_output.normal_finished:
        response = Response(status_code=400)
        # response.headers["is_finished"] = str(final_output.normal_finished)
        return response

    # Store result in server
    image_name = f"{request_id}.png"
    path = "./outputs/imgs/" + image_name
    final_output.output.images.save(path)

    response =  FileResponse(path, media_type="image/png")
    response.headers["image_name"] = image_name
    response.headers["is_finished"] = str(final_output.normal_finished)

    return response


@app.post("/clear")
async def clear(request: Request) -> Response:
    """Clear data and ready to release."""
    await engine.cler(
    sys.stdout.flush()
    sys.stderr.flush()
    return Response(status_code=200)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_args_to_parser(parser)

    args = parser.parse_args()

    host = args.__dict__.pop("host")
    port = args.__dict__.pop("port")

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncEngine.from_engine_args(engine_args)

    pipeline_cls: BasePipeline = get_pipeline_cls(engine.pipeline_config)
    sampling_param_cls: BaseSamplingParams = pipeline_cls.get_sampling_params_cls()

    uvicorn.run(app,
                host=host,
                port=port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)