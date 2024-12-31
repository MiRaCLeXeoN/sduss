import pandas as pd

from typing import Union, Optional, List, TYPE_CHECKING, Dict, Set, Iterable, Callable
from collections import defaultdict

from sduss.logger import init_logger

from ..wrappers import WorkerRequest, WorkerReqStatus

logger = init_logger(__name__)

class WorkerRequestPool:
    """Manager abstract of requests"""
    def __init__(
        self,
        support_resolutions: List[int],
    ):
        self.columns = ["request_id", "status", "resolution", "remain_steps"]
        self.requests = pd.DataFrame(columns=self.columns)
        self.requests.set_index("request_id", inplace=True)

        self.req_mapping: Dict[int, WorkerRequest] = {}

        self.support_resolutions = support_resolutions


    def add_requests(self, requests: Iterable[WorkerRequest]):
        """Add requests to request pool

        Args:
            requests (Iterable[WorkerRequest]): reqs

        Raises:
            RuntimeError: Requests should not already be tracked.
        """
        if isinstance(requests, WorkerRequest):
            requests = [requests]
        
        for req in requests:
            if req.request_id in self.req_mapping:
                raise RuntimeError(f"WorkerRequest with id {req.request_id} already exists.")
        
            # Add to mapping
            self.req_mapping[req.request_id] = req

            # Add to df
            self.requests.loc[req.request_id] = {
                "status": req.status,
                "resolution": req.sampling_params.resolution,
                "remain_steps": req.remain_steps,
            }


    def remove_requests(self, request_ids: Iterable[int]) -> List[WorkerRequest]:
        """Remove a request from the pool by its id."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]

        # Remove from dataframe
        self.requests.drop(request_ids, axis=0, inplace=True)
        # Remove from mapping
        reqs = []
        for req_id in request_ids:
            # Get the request
            req = self.req_mapping.pop(req_id)
            reqs.append(req)
        
        return reqs


    def get_by_ids(self, request_ids: Iterable[int]) -> List[WorkerRequest]:
        """Query a request by its id."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]

        reqs = []
        for req_id in request_ids:
            req = self.req_mapping.get(req_id)
            reqs.append(req)
            
        return reqs


    def get_ids_by_status(self, status: WorkerReqStatus):
        """Query all requests with a specific status."""
        return self.requests[self.requests["status"] == status].index.tolist()


    def get_ids_by_resolution(self, res: int):
        """Query all requests with a specific rank."""
        return self.requests[self.requests["resolution"] == res].index.tolist()
    
    
    def update_requests(self, requests: List[WorkerRequest]):
        """Update the status of a request and update references accordingly."""
        for req in requests:
            assert req.request_id in self.req_mapping

            # Update
            self.requests.at[req.request_id, "status"] = req.status
            self.requests.at[req.request_id, "remain_steps"] = req.remain_steps

            # The following colums don't change, no need to update
            # resolution


    def get_finished_req_ids(self) -> List[int]:
        req_ids = self.requests.index[self.requests["status"].apply(WorkerReqStatus.is_finished)].tolist()
        return req_ids
    
    
    def get_unfinished_req_ids(self) -> List[int]:
        req_ids = self.requests.index[~self.requests["status"].apply(WorkerReqStatus.is_finished)].tolist()
        return req_ids
    
    
    def get_unfinished_req_ids_by_res(self, resolution: int) -> List[int]:
        req_ids = self.requests.index[
            (~self.requests["status"].apply(WorkerReqStatus.is_finished)
             & self.requests["resolution"] == resolution)
        ].tolist()
        return req_ids
    
    
    def free_finished_reqs(self) -> List[WorkerRequest]:
        req_ids = self.get_finished_req_ids()
        return self.remove_requests(req_ids)
    
    
    def has_unfinished_reqs(self) -> bool:
        return (~self.requests["status"].apply(WorkerReqStatus.is_finished)).any()
    
    
    def get_unfinished_reqs(self) -> List[WorkerRequest]:
        req_ids = self.get_unfinished_req_ids()
        return self.get_by_ids(req_ids)

    
    def get_reqs_ids_by_complex(
        self,
        status: WorkerReqStatus = None,
        resolution: int = None,
        remain_steps: int = None
    ) -> List[int]:
        if (status is None
            and resolution is None
            and remain_steps is None):
            return []

        # Start with a default mask that includes all rows
        mask = pd.Series(True, index=self.requests.index)

        if status:
            mask = mask & (self.requests["status"] == status)
        if resolution:
            mask = mask & (self.requests["resolution"] == resolution)
        if remain_steps:
            mask = mask & (self.requests["remain_steps"] == remain_steps)
        
        return self.requests.index[mask].tolist()
    
    
    def get_req_ids_by_function(
        self,
        condition_func: Callable,
    ) -> List:
        """
        Filters requests based on a custom condition function.

        Args:
            condition_func (Callable): Functions that takes columns as inputs 
                and returns a boolean.

        Returns:
            List: A list of `request_id`s where the condition function evaluates to True.
        """
        # Apply the function row by row and get the mask
        mask = self.requests.apply(
            lambda row: condition_func(**{col_name:row[col_name] for col_name in self.columns}),
            axis=1
        )
        
        # Return the request IDs where the mask is True
        return self.requests.index[mask].tolist()
        
    
    def get_reqs_by_complex(
        self,
        status: WorkerReqStatus = None,
        resolution: int = None,
        remain_steps: int = None
    ) -> List[WorkerRequest]:
        req_ids = self.get_reqs_ids_by_complex(
            status,
            resolution,
            remain_steps,
        )
        return self.get_by_ids(req_ids)
    

    def get_log_status_str(self) -> str:
        return self.requests.to_string()
        
    
    def log_status(self):
        logger.debug(self.get_log_status_str())
        