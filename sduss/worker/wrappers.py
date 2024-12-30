import enum
import time

from typing import List, Dict, TYPE_CHECKING, Optional, Union

from .scheduler.esymred_utils import (DISCARD_SLACK, DENOISING_DDL, POSTPROCESSING_DDL, STANDALONE,
                            Hyper_Parameter)

if TYPE_CHECKING:
    from sduss.model_executor.sampling_params import BaseSamplingParams
    from sduss.model_executor.diffusers import BasePipelinePrepareOutput
    from sduss.model_executor.diffusers import BaseSchedulerStates
    from sduss.dispatcher import Request


class WorkerReqStatus(enum.IntEnum):
    """Status of a sequence."""
    # Empty
    EMPTY = enum.auto()             # No requests are scheduled
                                    # This is to handle condition where very last req is 
                                    # running post stage and resulting in no unfinished reqs
    # Running
    PREPARE = enum.auto()           # ready for prepare stage
    DENOISING = enum.auto()         # ready for denoising stage
    POSTPROCESSING = enum.auto()    # ready for postprocessing stage
    # Finished
    FINISHED_STOPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "WorkerReqStatus") -> bool:
        return status in [
            WorkerReqStatus.FINISHED_STOPPED,
            WorkerReqStatus.FINISHED_ABORTED,
        ]
    
    
    @staticmethod
    def is_normal_finished(status: "WorkerReqStatus") -> bool:
        return status in [
            WorkerReqStatus.FINISHED_STOPPED,
        ]


    @staticmethod
    def get_next_status(status: "WorkerReqStatus") -> Optional["WorkerReqStatus"]:
        if status == WorkerReqStatus.PREPARE:
            return WorkerReqStatus.DENOISING
        elif status == WorkerReqStatus.DENOISING:
            return WorkerReqStatus.POSTPROCESSING
        elif status == WorkerReqStatus.POSTPROCESSING:
            return WorkerReqStatus.FINISHED_STOPPED
        else:
            raise RuntimeError(f"We cannot decide next status for {status=}.")


    @staticmethod
    def get_finished_reason(status: "WorkerReqStatus") -> Union[str, None]:
        if status == WorkerReqStatus.FINISHED_STOPPED:
            finish_reason = "finished normally"
        elif status == WorkerReqStatus.FINISHED_ABORTED:
            finish_reason = "aborted"
        else:
            finish_reason = None
        return finish_reason


class WorkerRequest:
    """Schduler's request must be converted to worker's request."""
    def __init__(
        self,
        request_id: int,
        sampling_params: "BaseSamplingParams",
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.arrival_time = time.time()

        # Used by scheduler
        self.status = WorkerReqStatus.PREPARE
        self.remain_steps = sampling_params.num_inference_steps

        # Used by esymred
        self.start_denoising = False
        self.is_discard = False
        ## Predict time indicates the time estimated to run until complete all
        ## Unet iterations, with respect to the current workload.
        self.predict_time = None

        # Set when finishes
        self.finish_time = None
        self.output = None


    def is_finished(self):
        return WorkerReqStatus.is_finished(self.status)


    def update_predict_time(self, predict_time:float):
        self.predict_time = predict_time
    
    
    def abort(self):
        self.status = WorkerReqStatus.FINISHED_ABORTED
        self.finish_time = time.time()


    def set_slack(
        self, 
        model_name : str, 
        is_running : bool, 
        current_running_time_cost : float,
    ):
        # If discarded, return directly
        # TODO: Discard
        if self.is_discard:
            self.slack = DISCARD_SLACK
            return

        resolution = self.sampling_params.resolution
        status = self.status
        # Get ddl
        if status == WorkerReqStatus.WAITING or status == WorkerReqStatus.PREPARE:
            self.slack = 1e5
            return 
        elif status == WorkerReqStatus.DENOISING:
            stage = "denoising"
            ddl = DENOISING_DDL[model_name][str(resolution)]
        elif status == WorkerReqStatus.POSTPROCESSING:
            stage = "postprocessing"
            ddl = POSTPROCESSING_DDL[model_name][str(resolution)]
        
        unit_unet_time = STANDALONE[model_name][stage][str(resolution)]
        if stage == "postprocessing":
            self.slack = (ddl - unit_unet_time - current_running_time_cost - (time.time() - self.arrival_time)
                            ) / (unit_unet_time * Hyper_Parameter[model_name][stage][str(resolution)])
        elif stage == "denoising":
            # print(f"{ddl=},{unit_unet_time=},{self.predict_time},{current_running_time_cost=},{(time.time() - self.arrival_time)}")
            if is_running:
                # Suppose we have started at least one round
                self.slack = (ddl - self.predict_time - current_running_time_cost - (time.time() - self.arrival_time)
                                ) / unit_unet_time
            else:
                # Denoising not started yet
                self.slack = (ddl - unit_unet_time - current_running_time_cost - (time.time() - self.arrival_time)
                                ) / unit_unet_time
        self.remain_time = ddl - current_running_time_cost - (time.time() - self.arrival_time)

    
    def is_compatible_with(self, req: "Request") -> bool:
        return (self.status == req.status and
                self.sampling_params.is_compatible_with(req.sampling_params))

        
class WorkerOutput:
    def __init__(
        self,
        worker_reqs: List[WorkerRequest] = None,
    ) -> None:
        # Performance recording
        self.req_output_dict = {}
        for req in worker_reqs:
            self.req_output_dict[req.request_id] = req.output