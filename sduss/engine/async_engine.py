import asyncio
import time
import sys

from typing import (List, Any, Optional, TYPE_CHECKING, Type, Tuple, Dict,
                    AsyncGenerator, Set, Iterable, Union)
from functools import partial

from sduss.logger import init_logger
from sduss.config import (PipelineConfig, ParallelConfig, 
                          SchedulerConfig, EngineConfig)
from sduss.entrypoints.wrappers import ReqOutput
from sduss.executor.ray_executor import ray_initialize_cluster
from sduss.model_executor.sampling_params import BaseSamplingParams

from .arg_utils import AsyncEngineArgs
from .utils import AsyncEngineDeadError
from .async_engine_wrappers import _MpAsyncEngine

logger = init_logger(__name__)

def _raise_exception_on_finish(
        task: asyncio.Task,
        request_tracker: "RequestTracker"
    ) -> None:
    msg = ("Shielded task finished unexpectedly. This should never happen! "
           "You can open a issue on Github to report this bug.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutput for a request that can be
    iterated over asynchronously.

    Even though we return the generated image for only once,
    generator paradigm still works fine. 
    """
    def __init__(self, request_id: int) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False
    
    
    def put(self, item: ReqOutput) -> None:
        if self._finished:
            # We cannot put more items.
            return
        self._queue.put_nowait(item)
    
    
    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration)
        self._finished = True
    
    
    @property
    def finished(self) -> bool:
        return self._finished
    
    
    def __aiter__(self):
        return self
    
    
    async def __anext__(self) -> ReqOutput:
        result = await self._queue.get()
        if result is StopAsyncIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            # Unexpected exception caught. Raise it.
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""
    def __init__(self) -> None:
        self._request_streams_mapping: Dict[int, AsyncStream] = {}

        self._new_requests: asyncio.Queue[Tuple[AsyncStream, Dict]] = asyncio.Queue()

        self._finished_requests: asyncio.Queue[int] = asyncio.Queue()
        
        # To be assigned afterwards
        self.new_requests_event = None

    
    def __contains__(self, item):
        return item in self._request_streams_mapping
    
    
    def init_event(self) -> None:
        self.new_requests_event = asyncio.Event()
    
    
    def add_request(
            self,
            request_id: int,
            **engine_add_request_kwargs,
        ) -> None:
        """Add a request to be sent to the engine on
        the next background loop iteration."""
        if request_id in self._request_streams_mapping:
            raise ValueError(f"{request_id=} already exists.")
        
        stream = AsyncStream(request_id=request_id)
        # Add new request
        self._new_requests.put_nowait(
            (
                stream,
                {
                    "request_id": request_id,
                    **engine_add_request_kwargs,
                },
            )
        )
        # Set event
        self.new_requests_event.set()

        return stream
    
    
    def finish_request(self, request_id: int, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Abort request {request_id}.")

        # Add to finished queue
        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams_mapping or self._request_streams_mapping[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        # Finish request stream.
        self._request_streams_mapping[request_id].finish()
        self._request_streams_mapping.pop(request_id, None)

    
    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[int]]:
        """Get new and finished requests.
    
        self._request_streams_mapping is updated here.

        Returns:
            Tuple[List[Dict], Set[int]]: List of requests parameters and all
                finished requests' ids.
        """
        new_requests_params: List[Dict] = []
        # Use set to handle multi-cancelled requests
        finished_request_ids: Set[str] = set()

        # Remove finished reqs
        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_request_ids.add(request_id)
            self._request_streams_mapping.pop(request_id, None)

        # Start new reqs
        if not self._new_requests.empty():
            while not self._new_requests.empty():
                stream, new_request_param = self._new_requests.get_nowait()
                if stream.request_id in finished_request_ids:
                    # The request has already been aborted.
                    stream.finish()
                    continue
                # Keep a reference to stream
                self._request_streams_mapping[stream.request_id] = stream
                new_requests_params.append(new_request_param)

            self.new_requests_event.clear()

        return new_requests_params, finished_request_ids
    
    
    def process_request_output(
            self,
            request_output: ReqOutput,
            verbose: bool = False
        ) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        # self._request_streams_mapping[request_id].put(request_output)
        self._request_streams_mapping[request_id].put(request_output.normal_finished)
        # Abort if finished
        if request_output.is_finished:
            if request_output.normal_finished:
                logger.info(f"Finished request {request_id}.")
            elif not request_output.normal_finished:
                logger.info(f"Abort request {request_id}.")
            self.finish_request(request_id, verbose=False)
        else:
            logger.warning("Unexpected behavior: request output should be finished")

            
    def propagate_exception(
            self,
            exc: Exception,
            request_id: Optional[str] = None
        ) -> None:
        """Propagate an exception to request streams (all if request_id is None)."""
        if request_id is not None:
            self._request_streams_mapping[request_id].put(exc)
        else:
            # Propage exception to all streams
            for stream in self._request_streams_mapping.values():
                stream.put(exc)

    
    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self._request_streams_mapping) > 0


class AsyncEngine:

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        *args,
        start_engine_loop: bool = True,
        **kwargs,
    ) -> None:
        # Params
        self.start_engine_loop = start_engine_loop
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.engine_config = engine_config

        # Engine
        self.engine = self._init_engine(
            pipeline_config=pipeline_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            engine_config=engine_config,
            **kwargs
        )

        self.engine.execute_method_sync("engine_is_ready")

        # Asyncio loop
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage collected
        self._background_loop_unshielded = None
        self.background_loop: asyncio.Future = None

        self._request_tracker = RequestTracker()

        # if self.start_engine_loop:
        #     self.start_background_loop()

        if self.engine_config.log_requests:
            logger.info("AsyncEngine initialization done. System Ready.")
            sys.stderr.flush()
            sys.stdout.flush()


    def _init_engine(self, *args, **kwargs) -> 'Union[_MpAsyncEngine]':
        engine_class = _MpAsyncEngine
        return engine_class(*args, **kwargs)
        

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None 
                and not self.background_loop.done())


    async def engine_step(
        self,
    ) -> bool:
        """Kick the engine to process new arrived requests.

        Returns:
            bool: True if there are in-progress requests.
        """
        new_requsts_params, finished_request_ids = (
            self._request_tracker.get_new_and_finished_requests())
        
        # Add requests
        if len(new_requsts_params) > 0:
            await self.engine.execute_method_async("add_requests", False, new_requsts_params)
        
        # Peek at output if there is any
        request_outputs, has_unfinished_reqs = await self.engine.execute_method_async("step", True)
            
        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.engine_config.log_requests)

        return has_unfinished_reqs
    

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            # Try to cede control to other coroutines, like adding new reqs
            await asyncio.sleep(0.05)

    
    def start_background_loop(self) -> None:
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
        logger.info("Background loop initialization done.")
    
    
    async def add_request(
        self,
        request_id: int,
        sampling_params: BaseSamplingParams,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.engine_config.log_requests:
            logger.info(f"Received new request {request_id}")
        
        if not self.is_running:
            self.start_background_loop()

        stream = self._request_tracker.add_request(
            request_id=request_id,
            sampling_params=sampling_params,
            arrival_time=arrival_time,
        )

        return stream
    
    
    async def generate(
        self,
        request_id: int,
        sampling_params: BaseSamplingParams,
    ) -> AsyncGenerator:
        """Process a request.

        Args:
            request_id (int): Request ID.
            sampling_params (BaseSamplingParams): Sampling params.
        """

        arrival_time = time.time()

        try:
            stream = await self.add_request(request_id,
                                            sampling_params,
                                            arrival_time=arrival_time)
            # Step over async generator to get result
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the request.
            self.abort_requests([request_id])
            raise e
        

    @classmethod 
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True
    ) -> "AsyncEngine":
        """Creates an AsyncEngine from the engine arguments."""

        # Create the engine configs.
        (pipeline_config, parallel_config, scheduler_config, 
         engine_config) = engine_args.create_engine_configs()

        # Create the async LLM engine.
        engine = cls(
            pipeline_config,
            parallel_config,
            scheduler_config,
            engine_config,
            start_engine_loop=start_engine_loop,
        )
        return engine
    
    
    async def abort_requests(self, request_ids: List[int]) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")
        
        if not isinstance(request_ids, list):
            request_ids = [request_ids]

        await self.engine.execute_method_async("abort_requests", False, request_ids)


    async def clear(self):
        while self._request_tracker.has_unfinished_requests():
            # Wait for all requests to finish
            await asyncio.sleep(3)
        await self.engine.execute_method_async("clear", True)