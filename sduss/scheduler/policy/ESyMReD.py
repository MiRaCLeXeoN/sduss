import time
import joblib
import os

from typing import List, TYPE_CHECKING, Dict, Tuple

from sduss.scheduler.wrappers import ResolutionRequestQueue
from sduss.utils import get_os_env

from .policy import Policy
from ..wrappers import SchedulerOutput, RequestStatus
from ..utils import find_gcd
from ..esymred_utils import Hyper_Parameter

if TYPE_CHECKING:
    from sduss.scheduler import Request

class Predictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_path = model_path
        if "sd1.5" in model_path:
            self.model.gamma = 5*1e-4
            self.model.C = 2 * 1e5
        else:
            self.model.gamma = 2*1e-4
            self.model.C = 1e6
    
    def predict(self, task_distribute: List):
        return self.model.predict(task_distribute)
    
    def re_load(self):
        self.model = joblib.load(self.model_path)
        if "sd1.5" in self.model_path:
            self.model.gamma = 5*1e-4
            self.model.C = 2 * 1e5
        else:
            self.model.gamma = 2*1e-4
            self.model.C = 1e6

    def re_train(self, task_dataset: List, label_dataset: List):
        self.model.fit(task_dataset[-150:], label_dataset[-150:])
        joblib.dump(self.model, self.model_path)


class ESyMReD_Scheduler(Policy):
    """
    Features:
        Support:
            1. Dynamic-batching
            2. Mixed-precision scheudling
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resolution_list = kwargs.pop("support_resolutions")

        model_path: str = get_os_env("ESYMRED_PREDICTOR_PATH", check_none=True)
        postprocessing_exec_time_dir: str = get_os_env("ESYMRED_EXEC_TIME_DIR", check_none=True)

        if "sd1.5" in model_path:
            self.model_name = "sd1.5"
        else:
            self.model_name = "sdxl"

        self.predictor = Predictor(model_path)
        self.postprocessing_exec_time: Dict[str, Dict[str, List[float]]] = {
            "sd1.5": {},
            "sdxl": {}
        }
        self._get_postprocessing_time(postprocessing_exec_time_dir)
        
        # Hyper parameters
        self.postprocessing_ratio = Hyper_Parameter["postprocessing_ratio"]
        
        # predict_time indicates the estimated time for next round
        self.predict_time = 0
        self.finish_all_reqs = False
    
    
    def _get_postprocessing_time(self, dir_pth: str) -> None:
        for resolution in self.resolution_list:
            path = os.path.join(dir_pth, f"sm_util_{self.model_name}_{resolution}.csv")
            self.postprocessing_exec_time[self.model_name][str(resolution)] = list()
            with open(path, "r") as f:
                first_line = True
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if first_line:
                        first_line = False
                        continue
                    elements = line.strip().split(",")
                    self.postprocessing_exec_time[self.model_name][str(resolution)].append(float(elements[-1]))
        

    def _flatten_all_reqs(self) -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_unfinished_reqs())
        return reqs
    
    
    def _flatten_all_reqs_without_status(self, status: "RequestStatus") -> List['Request']:
        reqs = self._flatten_all_reqs()
        reqs = [req for req in reqs if req.status != status]
        return reqs
    
    def _get_all_reqs_by_status(self, status: "RequestStatus") -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs_by_status(status))
        return reqs
    
    def _get_all_finished_reqs(self) -> int:
        total_finished_num = 0
        for resolution_queue in self.request_pool.values():
            total_finished_num += resolution_queue.get_num_finished_reqs()
            total_finished_num += len(resolution_queue.get_queue_by_status(RequestStatus.FINISHED_ABORTED))
        return total_finished_num
    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        if self._get_all_finished_reqs() == 100:
            self.finish_all_reqs = True
            for resolution_queue in self.request_pool.values():
                resolution_queue.recover_aborted_requests()
        
        flattened_reqs = self._flatten_all_reqs_without_status(RequestStatus.WAITING)

        # If not reqs to schedule, return EMPTY
        if len(flattened_reqs) == 0:
            return SchedulerOutput(
                scheduled_requests={},
                status=RequestStatus.EMPTY,
                update_all_waiting_reqs=True,
            )

        if self.finish_all_reqs:
            if len(flattened_reqs) != 0:
                now = time.time()
                flattened_reqs.sort(key = lambda req: now - req.arrival_time, reverse=True)
                target_req = flattened_reqs[0]
                target_status = target_req.status
                queue = self._get_all_reqs_by_status(target_status)
                queue.sort(key=lambda req: now - req.arrival_time, reverse=True)
                res_reqs_dict: Dict[int, Dict[int, Request]] = {}
                num_to_collect = max_num
                best_latency_bs = 40
                while num_to_collect > 0 and queue and best_latency_bs > 0:
                    req = queue.pop(0)
                    res = req.sampling_params.resolution
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        res_reqs_dict[res][req.request_id] = req
                    num_to_collect -= 1
                    best_latency_bs -= (res // 256) ** 2
                is_sliced = None
                patch_size = None
                if target_status == RequestStatus.DENOISING:
                    if len(res_reqs_dict) > 1:
                        is_sliced = True
                        patch_size = find_gcd(list(res_reqs_dict))
                    else:
                        is_sliced = False
                        patch_size = list(res_reqs_dict.keys())[0]
                return SchedulerOutput(
                    scheduled_requests=res_reqs_dict,
                    status=target_status,
                    is_sliced=is_sliced,
                    patch_size=patch_size,
                )
        # Set slack
        for req in flattened_reqs:
            req.set_slack(self.model_name, False, self.predict_time)

        # Find the highest priority req
        now = time.time()
        index = 0
        flattened_reqs.sort(key = lambda req: req.slack, reverse=False)
        target_req = flattened_reqs[index]
        target_res = target_req.sampling_params.resolution
        target_status = target_req.status

        res_reqs_dict: Dict[int, Dict[int, Request]] = {}
        req_ids_to_abort: List[int] = []
        is_sliced = None
        patch_size = None
        if target_status == RequestStatus.PREPARE:
            queue = self._get_all_reqs_by_status(target_status)
            queue.sort(key=lambda req: now - req.arrival_time, reverse=True)
            num_to_collect = max_num

            while num_to_collect > 0 and queue:
                req = queue.pop(0)
                res = req.sampling_params.resolution
                if res not in res_reqs_dict:
                    res_reqs_dict[res] = {req.request_id : req}
                else:
                    res_reqs_dict[res][req.request_id] = req
                num_to_collect -= 1
        elif target_status == RequestStatus.POSTPROCESSING:
            queue = self._get_all_reqs_by_status(target_status)
            queue.sort(key=lambda req: now - req.arrival_time, reverse=True)
            num_to_collect = max_num

            while num_to_collect > 0 and queue:
                req = queue.pop(0)
                res = req.sampling_params.resolution
                if res != target_res:
                    # We add only one resolution for postprocessing stage
                    # ! FIXME: We can send multiple resolutions, but this may ruin the predict_time,
                    # ! since dependent data are collected with only single resolution.
                    continue
                elif res not in res_reqs_dict:
                    res_reqs_dict[res] = {req.request_id : req}
                else:
                    # We only add reqs that can satisfy SLO
                    # ! FIXME: `target_req.remain_time` or `req.remain_time` ?
                    if (self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])] 
                            / target_req.remain_time) < self.postprocessing_ratio:
                        res_reqs_dict[res][req.request_id] = req
                        self.predict_time = self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])]
                    else:
                        # ! FIXME: Can we abort remaining unselected reqs?
                        break
                num_to_collect -= 1
        elif target_status == RequestStatus.DENOISING:
            running_reqs_list: List['Request'] = list()
            # Pick up running denoising reqs
            for req in flattened_reqs:
                if req.start_denoising and req.status == RequestStatus.DENOISING:
                    res = req.sampling_params.resolution
                    # Add reqs that are currently running. They are destined to continue running.
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        res_reqs_dict[res][req.request_id] = req
                    running_reqs_list.append(req)
            
            num_to_collect = max_num
            while True:
                # Add new requests to start denoising
                if not target_req.start_denoising:
                    if len(running_reqs_list) > 0:
                        # Collect number of activated reqs by resolution.
                        res_candidate_dict: Dict[int, int] = {res : len(res_reqs_dict[res]) for res in res_reqs_dict}
                        for res in self.resolution_list:
                            if res not in res_candidate_dict:
                                res_candidate_dict[res] = 0

                        # Choose the resolution that can maximize throughput
                        # ! This is actually choosing the minimal resolution that has unfinifhsed reqs
                        best_tp_res = None
                        for res in self.resolution_list:
                            reqs_list = self.request_pool[res].get_all_unfinished_reqs()
                            if len(reqs_list) != 0:
                                find_best_tp_res = False
                                for req in reqs_list:
                                    if req.status == RequestStatus.DENOISING and not req.start_denoising:
                                        best_tp_res = res
                                        find_best_tp_res = True
                                        break
                                if find_best_tp_res:
                                    break
                        
                        # Get distributions
                        running_reqs_list.sort(key = lambda req: req.remain_steps, reverse=False)
                        task_distribute_original, task_distribute_slo, task_distribute_best_tp = (
                            self._get_distributions(
                                res_candidate_dict=res_candidate_dict,
                                target_res=target_res,
                                best_tp_res=best_tp_res,
                                target_req=target_req,
                                running_reqs_list=running_reqs_list,
                            ))

                        # Predict time for each distribution
                        predict_time_original = self.predictor.predict(task_distribute_original)
                        predict_time_slo = self.predictor.predict(task_distribute_slo)
                        predict_time_best_tp = self.predictor.predict(task_distribute_best_tp)
                        
                        spend_time, spend_time_original, spend_time_best_tp, target_req_pred_time = (
                            self._get_spend_time(
                                predict_time_original=predict_time_original,
                                predict_time_slo=predict_time_slo,
                                predict_time_best_tp=predict_time_best_tp,
                                target_req=target_req,
                                running_reqs_list=running_reqs_list,
                            ))

                        target_req.update_predict_time(target_req_pred_time)
                        target_req.set_slack(self.model_name, True, self.predict_time) 

                        running_reqs_list.sort(key = lambda req: req.slack, reverse=False)
                        # If we have low slack req, i.e., very urgent one, we don't add more reqs.
                        # TODO: A Hyper parameter here
                        if running_reqs_list[0].slack < 0.1:
                            break

                        if target_req.slack < 0:
                            target_req.status = RequestStatus.FINISHED_ABORTED
                            req_ids_to_abort.append(target_req.request_id)
                        else:
                            # 如果最紧急的请求依然不是特别紧急，则追求最大吞吐量，修改target_req为最小resolution的request
                            if target_req.slack > 1:
                                is_get_best_tp = True
                                for i in range(index, len(flattened_reqs)):
                                    req = flattened_reqs[i]
                                    if (req.sampling_params.resolution == best_tp_res 
                                        and req.status == RequestStatus.DENOISING and not req.start_denoising):
                                        target_req = req
                                        target_res = best_tp_res
                                        break
                            else:
                                is_get_best_tp = False
                            # If some reqs are urgent, we prioritize satifying SLO.
                            if not is_get_best_tp:
                                if target_res not in res_reqs_dict:
                                    res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                else:
                                    res_reqs_dict[target_res][target_req.request_id] = target_req
                                target_req.start_denoising = True
                                num_to_collect -= 1
                                running_reqs_list.append(target_req)
                                self.predict_time = predict_time_slo[0]
                            else:
                                # Add this req for best throughput is it won't significantly 
                                # TODO: A Hyper parameter here
                                # ! FIXEME: Should we multiply the estimated time consumtion of solely running
                                # ! target_req by its remain_steps in the following `if` statement?
                                if spend_time_best_tp / (spend_time_original + self.predictor.predict(
                                    [[1 if res == target_res else 0 for res in self.resolution_list]])[0]) < 0.95:
                                    if target_res not in res_reqs_dict:
                                        res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                    else:
                                        res_reqs_dict[target_res][target_req.request_id] = target_req
                                    target_req.start_denoising = True
                                    running_reqs_list.append(target_req)
                                    num_to_collect -= 1
                                    self.predict_time = predict_time_best_tp[0]
                                else:
                                    break
                    # end if len(running_reqs_list) > 0:
                    else:
                        # No currently activated reqs
                        # So add current req directly
                        task_distribute = list()
                        for res in self.resolution_list:
                            if res == target_res:
                                task_distribute.append(1)
                            else:
                                task_distribute.append(0)
                        if target_res not in res_reqs_dict:
                            res_reqs_dict[target_res] = {target_req.request_id : target_req}
                        else:
                            res_reqs_dict[target_res][target_req.request_id] = target_req
                        target_req.start_denoising = True
                        running_reqs_list.append(target_req)
                        num_to_collect -= 1
                        self.predict_time = self.predictor.predict(task_distribute)
                # end if not target.start_denoising

                index += 1
                if index < len(flattened_reqs):
                    target_req = flattened_reqs[index]
                    target_res = target_req.sampling_params.resolution
                    target_status = target_req.status
                    # We don't add more denoising reqs that are preceded by POSTPROCESSING
                    # or PREPARE reqs, to prevent those reqs from expiration.
                    if target_status != RequestStatus.DENOISING:
                        target_status = RequestStatus.DENOISING
                        break
                else:
                    break
                if num_to_collect <= 0:
                    break
            # end while

            if len(res_reqs_dict) > 1:
                is_sliced = True
                patch_size = find_gcd(list(res_reqs_dict))
            else:
                is_sliced = False
                patch_size = list(res_reqs_dict.keys())[0]
            
        return SchedulerOutput(
            scheduled_requests=res_reqs_dict,
            status=target_status,
            abort_req_ids=req_ids_to_abort,
            is_sliced=is_sliced,
            patch_size=patch_size,
            update_all_waiting_reqs=True,
        )
    
    
    def _get_distributions(
        self,
        res_candidate_dict: Dict[int, int],
        target_res: int,
        best_tp_res: int,
        target_req: 'Request',
        running_reqs_list: List['Request'],
    ) -> Tuple[List, List, List]:
        # task_distribute_original: current distribution 
        # task_distribute_slo: distribution if the most urgent reqs is added
        # task_distribute_best_tp: distribution if req that can maximize tp is added
        task_distribute_original: List[List[int]] = list()
        task_distribute_original.append([res_candidate_dict[res] for res in self.resolution_list])
        task_distribute_slo: List[List[int]] = list()
        task_distribute_slo.append([res_candidate_dict[res] if res != target_res else (
            res_candidate_dict[res] + 1) for res in self.resolution_list])
        task_distribute_best_tp: List[List[int]] = list()
        task_distribute_best_tp.append([res_candidate_dict[res] if res != best_tp_res else (
            res_candidate_dict[res] + 1) for res in self.resolution_list])
        
        # 遍历待执行任务列表，计算每个任务在三种情况下的理论执行时间
        remain_steps = running_reqs_list[0].remain_steps
        res_candidate_dict[running_reqs_list[0].sampling_params.resolution] -= 1
        finish_tasks = 1
        while finish_tasks < len(running_reqs_list):
            req = running_reqs_list[finish_tasks]
            # 如果剩余的任务的remain_step与之前不同，说明在req.remain_steps - remain_steps步之内，
            # 执行的任务分布没有发生变化，直到这个任务也结束执行denoising
            if req.remain_steps != remain_steps:
                task_distribute_original.append([res_candidate_dict[res] for res in self.resolution_list])
                task_distribute_slo.append([res_candidate_dict[res] if res != target_res else (
                    res_candidate_dict[res] + 1) for res in self.resolution_list])
                task_distribute_best_tp.append([res_candidate_dict[res] if res != best_tp_res else (
                    res_candidate_dict[res] + 1) for res in self.resolution_list])
            res_candidate_dict[req.sampling_params.resolution] -= 1
            remain_steps = req.remain_steps
            finish_tasks += 1
        # task_distribute.append([resolution_candidate_list[res] for res in self.resolution_list])
        # 如果激活队列中也存在还未执行的任务，则无需额外处理，否则，需要计算单独执行待添加的任务需要执行的时间
        if remain_steps != target_req.remain_steps:
            task_distribute_slo.append([res_candidate_dict[res] if res != target_res else (
                res_candidate_dict[res] + 1) for res in self.resolution_list])
            task_distribute_best_tp.append([res_candidate_dict[res] if res != best_tp_res else (
                res_candidate_dict[res] + 1) for res in self.resolution_list])
    
    
    def _get_spend_time(
        self,
        predict_time_original: List[float],
        predict_time_slo: List[float],
        predict_time_best_tp: List[float],
        target_req: 'Request',
        running_reqs_list: List['Request'],
    ) -> Tuple[float, float, float, float]:
        spend_time = predict_time_slo[0] * remain_steps
        # 统计原激活队列与新加最大吞吐量的情况的预测执行时间
        spend_time_original = predict_time_original[0] * remain_steps
        spend_time_best_tp = predict_time_best_tp[0] * remain_steps
        running_reqs_list[0].update_predict_time(spend_time)
        # ! FIXME: Is self.predict correct at the first denoising round of first req?
        running_reqs_list[0].set_slack(self.model_name, True, self.predict_time)
        predict_index = 0
        finish_tasks = 1
        remain_steps = running_reqs_list[0].remain_steps
        # 统计添加最紧急的任务的情况下，各个任务的预计完成时间
        while finish_tasks < len(running_reqs_list):
            req = running_reqs_list[finish_tasks]
            if req.remain_steps != remain_steps:
                predict_index += 1
                # update_steps = req.remain_steps - remain_steps
                spend_time += (req.remain_steps - remain_steps) * predict_time_slo[predict_index]
                spend_time_original += (req.remain_steps - remain_steps) * predict_time_original[predict_index]
                spend_time_best_tp += (req.remain_steps - remain_steps) * predict_time_best_tp[predict_index]
            req.update_predict_time(spend_time_original)
            req.set_slack(self.model_name, True, self.predict_time)
            remain_steps = req.remain_steps
            finish_tasks += 1
        spend_time_best_tp += (target_req.remain_steps - remain_steps) * predict_time_best_tp[-1]
        target_req_pred_time = (target_req.remain_steps - remain_steps) * predict_time_slo[-1] + spend_time

        return (spend_time, spend_time_original, spend_time_best_tp, target_req_pred_time)