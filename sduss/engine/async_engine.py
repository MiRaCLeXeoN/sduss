import asyncio

from typing import List, Any
from functools import partial

from sduss.logger import init_logger
from sduss.entrypoints.outputs import RequestOutput

from .engine import Engine

class _AsyncEngine(Engine):
    
    async def step_async(self) -> List[RequestOutput]:
        pass

    
    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method asynchrously on all workers."""
        coroutines = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coroutines.append(worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coroutines.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))
        
        all_outputs = await asyncio.gather(*coroutines)

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output