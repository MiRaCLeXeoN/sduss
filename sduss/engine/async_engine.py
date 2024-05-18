import asyncio
import time
import sys

import ray

from typing import (List, Any, Optional, TYPE_CHECKING, Type, Tuple, Dict,
                    AsyncGenerator, Set, Iterable, Union)
from functools import partial
from datetime import datetime

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sduss.logger import init_logger
from sduss.config import (PipelineConfig, ParallelConfig, 
                          SchedulerConfig, EngineConfig)
from sduss.entrypoints.outputs import RequestOutput
from sduss.worker import WorkerOutput
from sduss.executor.ray_executor import initialize_cluster
from sduss.model_executor.sampling_params import BaseSamplingParams
from sduss.scheduler import RequestStatus

from .engine import Engine
from .arg_utils import AsyncEngineArgs
from .utils import AsyncEngineDeadError

if TYPE_CHECKING:
    import ray
    from ray.util.placement_group import PlacementGroup
    from sduss.executor.ray_executor import RayExecutor

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
    
    
    def put(self, item: RequestOutput) -> None:
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
    
    
    async def __anext__(self) -> RequestOutput:
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
    
    
    def abort_request(self, request_id: int, verbose: bool = False) -> None:
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

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_request_ids.add(request_id)
            self._request_streams_mapping.pop(request_id, None)

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
            request_output: RequestOutput,
            verbose: bool = False
        ) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams_mapping[request_id].put(request_output)
        # Abort if finished
        if request_output.is_finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

            
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


class _AsyncEngine(Engine):
    
    # TODO: Since we enforced engine using ray, this is unnecessary.
    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.
        """
        # TODO: Incorrect
        if self.engine_config.non_blocking_step:
            return await self._step_nonblocking_async()
        else:
            return await self._step_blocking_async()
    
    
    async def _step_blocking_async(self) -> List[RequestOutput]:
        """Performs one denoising iteration and returns newly generated results."""
        scheduler_output, req_ids = self._schedule()

        output = None
        if scheduler_output.status == RequestStatus.WAITING:
            # Currently, we don't do anything in waiting stage
            pass
        elif scheduler_output.status == RequestStatus.PREPARE:
            # For prepare stage inference
            # TODO(MX): We may pass schduler output directly
            output: WorkerOutput = await self._run_workers_blocking_async(
                "exec_prepare_stage", 
                self.workers,
                scheduler_reqs=scheduler_output.get_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)
        elif scheduler_output.status == RequestStatus.DENOISING:
            # For denoising stage inference
            await self._run_workers_blocking_async(
                "exec_denoising_stage", 
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                is_sliced=scheduler_output.is_sliced,
                patch_size=scheduler_output.patch_size)
        elif (scheduler_output.status == RequestStatus.POSTPROCESSING):
            # For post stage inference
            output: WorkerOutput = await self._run_workers_blocking_async(
                "exec_post_stage",
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
            )
        else:
            raise RuntimeError(f"Unexpected status {scheduler_output.status}.")
        
        output = self._process_output(scheduler_output=scheduler_output,
                                       req_ids=req_ids,
                                       output=output,)

        if self.engine_config.log_status:
            self._log_system_states(scheduler_output)
        
        return output
    
    
    async def _step_nonblocking_async(self):
        """Non-blocking step."""
        # 1. Schedule
        scheduler_output, req_ids = self._schedule()
        
        if self.engine_config.log_status:
            self._log_system_states(scheduler_output)

        # 2. Wait for result from previous round
        # This must be after step 1 to truly overlap scheduling and execution.
        prepare_output, denoising_output, postprocessing_output = (
            await self.get_prev_handlers_output_async(get_output_all_workers=False))

        # 3. Schedule prepare if prepare reqs available
        if scheduler_output.has_prepare_requests():
            # We don't expect prepare stage if we have overlapped prepare-requests to process
            assert scheduler_output.status != RequestStatus.PREPARE
            self.prev_prepare_handlers = await self._run_workers_nonblocking_async(
                "exec_prepare_stage", 
                self.prepare_workers,
                scheduler_reqs=scheduler_output.get_prepare_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)

        # 4. Issue tasks to workers
        if scheduler_output.status == RequestStatus.WAITING:
            # Currently, we don't do anything in waiting stage
            if prepare_output is not None:
                # We don't need to preserve the handlers
                await self._run_workers_nonblocking_async(
                    "receive_prepare_output",
                    self.workers,
                    prepare_output=prepare_output,
                )
        elif scheduler_output.status == RequestStatus.PREPARE:
            # Only when there is no denoising or postprocessing reqs running will
            # prepare stage be scheduled.
            if prepare_output is not None:
                # We don't need to preserve the handlers
                await self._run_workers_nonblocking_async(
                    "receive_prepare_output",
                    self.workers,
                    prepare_output=prepare_output,
                )
            # Requests are derived from normal reqs instead of prepare_reqs in shceduler_output
            self.prev_prepare_handlers = await self._run_workers_nonblocking_async(
                "exec_prepare_stage",
                self.prepare_workers,
                scheduler_reqs=scheduler_output.get_reqs_as_list(),
                use_mixed_precision=self.scheduler_config.use_mixed_precision)
        elif scheduler_output.status == RequestStatus.DENOISING:
            # For denoising stage inference
            # transfer prepare result from previous round to worker
            self.prev_denoising_handlers = await self._run_workers_nonblocking_async(
                "exec_denoising_stage", 
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                is_sliced=scheduler_output.is_sliced,
                patch_size=scheduler_output.patch_size,
                prepare_output=prepare_output,)
        elif scheduler_output.status == RequestStatus.POSTPROCESSING:
            # For post stage inference
            self.prev_postprocessing_handlers = await self._run_workers_nonblocking_async(
                "exec_post_stage",
                self.workers,
                req_ids=req_ids,
                use_mixed_precision=self.scheduler_config.use_mixed_precision,
                prepare_output=prepare_output,
            )
        else:
            raise RuntimeError(f"Unexpected status {scheduler_output.status}.")
        
        # 5. Process output and update requests status.
        output = self._process_nonblocking_output(scheduler_output=scheduler_output,
                                                  req_ids=req_ids,
                                                  prepare_output=prepare_output,
                                                  denoising_output=denoising_output,
                                                  postprocessing_output=postprocessing_output,)
        
        return output

    
    async def _run_workers_blocking_async(
        self,
        method: str,
        workers: List['RayExecutor'],
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method asynchrously on all workers.
        Blocking means: for current coroutine, execution will be blocked here.
        Async means: for the whole process, this function is executed async.
        """
        coroutines = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coroutines.append(worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coroutines.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))
        
        # We must use asyncio's `await` to enable coroutine's switching.
        # If we use ray.get, event loop will get blocked here.
        all_outputs = await asyncio.gather(*coroutines)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
    
    
    async def _run_workers_nonblocking_async(
        self,
        method: str,
        workers: List['RayExecutor'],
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method asynchrously on all workers.
        Non-blocking means: for current coroutine, execution won't be blocked inside.
        Async means: for the whole process, this function is executed async.

        Essentially, this does the same as the synchronous version.
        """
        # TODO: We may reuse `_run_workers_nonblocking`
        assert self.parallel_config.worker_use_ray, "Only ray workers supports non blocking calls."
        obj_refs = []
        for worker in workers:
            executor = partial(worker.execute_method.remote, method)
            obj_ref = executor(*args, **kwargs)
            obj_refs.append(obj_ref)
        return obj_refs
    
    
    async def get_prev_handlers_output_async(
        self,
        get_output_all_workers: bool = False,
    ):
        """Get output from handlers set by previous round asynchronously.

        Args:
            get_output_all_workers (bool, optional): If true, outputs from all workers
                will be returned as a list. Otherwise only the first output will be extracted
                and returned.
        """
        prepare_output = denoising_output = postprocessing_output = None
        if self.prev_prepare_handlers:
            prepare_output = await asyncio.gather(*self.prev_prepare_handlers)
            self.prev_prepare_handlers = None
        if self.prev_denoising_handlers:
            denoising_output = await asyncio.gather(*self.prev_denoising_handlers)
            self.prev_denoising_handlers = None
        if self.prev_postprocessing_handlers:
            postprocessing_output = await asyncio.gather(*self.prev_postprocessing_handlers)
            self.prev_postprocessing_handlers = None
        
        if get_output_all_workers:
            return prepare_output, denoising_output, postprocessing_output

        prepare_output = prepare_output[0] if prepare_output else prepare_output
        denoising_output = denoising_output[0] if denoising_output else denoising_output
        postprocessing_output = postprocessing_output[0] if postprocessing_output else postprocessing_output

        return prepare_output, denoising_output, postprocessing_output
    
    
class AsyncEngine:

    _engine_class: Type[_AsyncEngine] = _AsyncEngine
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        distributed_init_method: str,
        gpu_pg: Optional["PlacementGroup"],
        cpu_pg: Optional["PlacementGroup"],
        engine_use_ray: bool,
        worker_use_ray: bool,
        *args,
        start_engine_loop: bool = True,
        **kwargs,
    ) -> None:
        # Params
        self.engine_use_ray = engine_use_ray
        self.worker_use_ray = worker_use_ray
        self.start_engine_loop = start_engine_loop
        self.pipeline_config = pipeline_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.engine_config = engine_config

        assert isinstance(engine_config, EngineConfig)
        # Engine
        self.engine = self._init_engine(
            pipeline_config=pipeline_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            engine_config=engine_config,
            distributed_init_method=distributed_init_method,
            gpu_pg=gpu_pg,
            cpu_pg=cpu_pg,
        )

        if self.engine_config.engine_use_ray:
            ray.get(self.engine.engine_is_ready.remote())

        # Asyncio loop
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage collected
        self._background_loop_unshielded = None
        self.background_loop: asyncio.Future = None

        self._request_tracker = RequestTracker()

        if self.engine_config.log_requests:
            logger.info("AsyncEngine initialization done. System Ready.")
            sys.stderr.flush()
            sys.stdout.flush()


    def _init_engine(self, *args, **kwargs) -> Union[_AsyncEngine, "ray.ObjectRef"]:
        cpu_pg = kwargs["cpu_pg"]

        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            # We must use num_cpus=0 to allow free actor scheduling in ray.
            # This doesn't imply that we will not be allocated CPUs to.
            engine_class = ray.remote(
                                num_cpus=1,
                                num_gpus=0,
                                scheduling_strategy=PlacementGroupSchedulingStrategy(
                                    placement_group=cpu_pg,
                                    placement_group_bundle_index=0,
                                    placement_group_capture_child_tasks=True,
                                )
                            )(self._engine_class).remote
        else:
            raise RuntimeError(f"Currently, {self._engine_class.__name__} doesn't support "
                               f"a combination of {self.engine_use_ray=} and {self.worker_use_ray=}")
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
        
        if self.engine_config.engine_use_ray:
            await self.engine.add_request_batch.remote(new_requsts_params)
        else:
            self.engine.add_request_batch(new_requsts_params)
        
        # Finished requests are automatically released. Don't re-abort.
        # if finished_request_ids:
        #     await self._engine_abort_reqs(finished_request_ids)
        
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.engine_config.log_requests)

        if self.engine_use_ray:
            has_unfinished_reqs = await self.engine.has_unfinished_normal_requests.remote()
        else:
            has_unfinished_reqs = self.engine.has_unfinished_normal_requests()
        return has_unfinished_reqs
    

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            # Try to cede control to other coroutines
            await asyncio.sleep(0)

    
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
    
    
    async def add_request(
        self,
        request_id: int,
        sampling_params: BaseSamplingParams,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.engine_config.log_requests:
            logger.info(f"Received new request {request_id}")
        
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise RuntimeError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")
        
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
            self._abort_req(request_id)
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
        # Initialize the cluster.
        distributed_init_method, gpu_pg, cpu_pg = initialize_cluster(
            parallel_config, scheduler_config, engine_config.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(
            pipeline_config,
            parallel_config,
            scheduler_config,
            engine_config,
            distributed_init_method,
            gpu_pg,
            cpu_pg,
            engine_config.engine_use_ray,
            parallel_config.worker_use_ray,
            start_engine_loop=start_engine_loop,
        )
        return engine
    
    
    async def abort_req(self, request_id: int) -> None:
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

        return self._abort_req(request_id)

    
    def _abort_req(self, request_id: int) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.engine_config.log_requests)
        
    
    async def _engine_abort_reqs(self, request_ids: Iterable[int]):
        if self.engine_use_ray:
            await self.engine.abort_requests.remote(request_ids)
        else:
            self.engine.abort_requests(request_ids)

    
    async def clear(self):
        ray.get(self.engine.clear.remote())