import pandas as pd

from typing import Union, Optional, List, TYPE_CHECKING, Dict, Set, Iterable
from collections import defaultdict

from sduss.logger import init_logger

from .wrappers import ReqStatus, Request

logger = init_logger(__name__)

class RequestPool:
    """Manager abstract of requests"""
    def __init__(
        self,
        dp_size: int,
    ):
        self.requests = pd.DataFrame(columns=["request_id", "status", "dp_rank",
                                              "resolution"])
        self.requests.set_index("request_id", inplace=True)

        self.req_mapping: Dict[int, Request] = {}

        self.dp_size = dp_size


    def add_requests(self, requests: Iterable[Request]):
        """Add requests to request pool

        Args:
            requests (Iterable[Request]): reqs

        Raises:
            RuntimeError: Requests should not already be tracked.
        """
        if isinstance(requests, Request):
            requests = [requests]
        
        for req in requests:
            if req.request_id in self.req_mapping:
                raise RuntimeError(f"Request with id {req.request_id} already exists.")
        
            # Add to mapping
            self.req_mapping[req.request_id] = req

            # Add to df
            self.requests.loc[req.request_id] = {
                "status": req.status,
                "dp_rank": req.dp_rank,
                "resolution": req.sampling_params.resolution,
            }


    def remove_requests(self, request_ids: Iterable[int]) -> List[Request]:
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


    def get_by_ids(self, request_ids: Iterable[int]) -> List[Request]:
        """Query a request by its id."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]

        reqs = []
        for req_id in request_ids:
            req = self.req_mapping.get(req_id)
            reqs.append(req)
            
        return reqs


    def get_ids_by_status(self, status: ReqStatus):
        """Query all requests with a specific status."""
        return self.requests[self.requests["status"] == status].index.tolist()


    def get_ids_by_dp_rank(self, dp_rank: int):
        """Query all requests with a specific rank."""
        return self.requests[self.requests["dp_rank"] == dp_rank].index.tolist()
    
    
    def get_pixels_all_dp_rank(self):
        # apply takes a dataframe as its input!
        d = self.requests.groupby("dp_rank")["resolution"].apply(lambda df: (df ** 2).sum()).to_dict()
        # Pad for empty dp rank
        for i in range(self.dp_size):
            if i not in d:
                d[i] = 0
        return d
    
    
    def update_requests(self, requests: List[Request]):
        """Update the status of a request and update references accordingly."""
        for req in requests:
            assert req.request_id in self.req_mapping

            # Update
            self.requests.at[req.request_id, "status"] = req.status
            self.requests.at[req.request_id, "dp_rank"] = req.dp_rank

            # The following colums don't change, no need to update
            # resolution
    
    
    def has_unfinished_reqs(self) -> bool:
        return len(self.req_mapping) > 0
    
    
    def get_num_unfinished_reqs(self) -> bool:
        return len(self.req_mapping)

    
    def get_num_unfinished_reqs_by_dp_rank(self) -> Dict:
        d = self.requests["dp_rank"].value_counts().to_dict()
        for i in range(self.dp_size):
            if i not in d:
                d[i] = 0
        return d