from typing import Iterable, Dict, List, TYPE_CHECKING

from .wrappers import Request

if TYPE_CHECKING:
    from .wrappers import SchedulerOutputReqsType

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


def convert_list_to_res_dict(reqs: List[Request]) -> 'SchedulerOutputReqsType':
    res_dict: 'SchedulerOutputReqsType' = {}
    for req in reqs:
        res = req.sampling_params.resolution
        if res not in res_dict:
            res_dict[res] = {req.request_id : req}
        else:
            res_dict[res][req.request_id] = req
    return res_dict