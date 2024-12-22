import asyncio

from typing import TYPE_CHECKING, List, Any
from functools import partial

from sduss.entrypoints.outputs import RequestOutput
from sduss.scheduler import RequestStatus
from sduss.worker import WorkerOutput

from ..engine import Engine


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
        workers: List,
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
        workers: List,
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