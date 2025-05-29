from typing import Iterable, Dict, List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .wrappers import SchedulerOutputReqsType
    from ..wrappers import WorkerRequest

def find_gcd(resolutions: Iterable[int]):

    def euclid(x, y):
        while y:
            x, y = y, x % y
        return x

    cur_gcd = None
    for res in resolutions:
        if cur_gcd is None:
            cur_gcd = res
        else:
            cur_gcd = euclid(cur_gcd, res)
    return cur_gcd


def convert_list_to_res_dict(
        reqs: 'List[WorkerRequest]',
        num: Optional[int] = None,
    ) -> 'SchedulerOutputReqsType':
    if num is None:
        # TODO(MX): Use a parameter to represent INIFINITY
        num = 1e8

    res_dict: 'SchedulerOutputReqsType' = {}
    for req in reqs:
        if num <= 0:
            break
        res = req.sampling_params.resolution
        if res not in res_dict:
            res_dict[res] = {req.request_id : req}
        else:
            res_dict[res][req.request_id] = req
        num -= 1
    return res_dict