import time
import joblib
import os
import numpy as np
import time
from typing import List, TYPE_CHECKING, Dict, Tuple

from sduss.utils import get_os_env
from sduss.logger import init_logger

from .policy import Policy
from ..wrappers import SchedulerOutput, WorkerReqStatus
from ..utils import find_gcd, convert_list_to_res_dict
from ..esymred_utils import Hyper_Parameter

if TYPE_CHECKING:
    from sduss.dispatcher import Request

logger = init_logger(__name__)

class Predictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_path = model_path
        # if "sd1.5" in model_path:
        #     self.model.gamma = 5*1e-4
        #     self.model.C = 2 * 1e5
        # else:
        #     self.model.gamma = 2*1e-4
        #     self.model.C = 1e6
        if "sdxl" in model_path:
            self.latency = {
                "512": 66.4,
                "768": 76.6,
                "1024": 79.8
            }
    
    def get_latency(self, resolution):
        return self.latency[str(resolution)]

    def predict(self, task_distribute: List):
        task_distribute = np.array(task_distribute)
        if "sd1.5" in self.model_path:
            data = task_distribute[:, :1] + task_distribute[:, 1:2] * 4 + task_distribute[:, 2:3] * 9
        else:
            data = task_distribute[:, :1] * 4 + task_distribute[:, 1:2] * 9 + task_distribute[:, 2:3] * 16
        data = np.concatenate((data, np.expand_dims(np.count_nonzero(task_distribute, axis=1), axis=0).T), axis=1)
        return self.model.predict(data) / 1000
    
    # def re_load(self):
    #     self.model = joblib.load(self.model_path)
    #     if "sd1.5" in self.model_path:
    #         self.model.gamma = 5*1e-4
        #     self.model.C = 2 * 1e5
        # else:
        #     self.model.gamma = 2*1e-4
        #     self.model.C = 1e6

    # def re_train(self, task_dataset: List, label_dataset: List):
    #     self.model.fit(task_dataset[-150:], label_dataset[-150:])
    #     joblib.dump(self.model, self.model_path)


class ESyMReD_Scheduler(Policy):
    """
    Features:
        Support:
            1. Dynamic-batching
            2. Mixed-precision scheudling
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resolution_list = sorted(list(self.request_pool.keys()))

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
            reqs.extend(resolution_queue.get_all_unfinished_normal_reqs())
        return reqs
    
    
    def _flatten_all_reqs_without_status(self, status: "WorkerReqStatus") -> List['Request']:
        reqs = self._flatten_all_reqs()
        reqs = [req for req in reqs if req.status != status]
        return reqs

    
    def _get_all_reqs_by_status(self, status: "WorkerReqStatus") -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs_by_status(status))
        return reqs

    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration."""
        
        flattened_reqs = self._flatten_all_reqs_without_status(WorkerReqStatus.WAITING)

        # If not reqs to schedule, return EMPTY
        if len(flattened_reqs) == 0:
            return SchedulerOutput(
                scheduled_requests={},
                status=WorkerReqStatus.EMPTY,
                update_all_waiting_reqs=True,
            )
        
        # complete request as soon as we can
        post_processing_queue = self._get_all_reqs_by_status(RequestStatus.POSTPROCESSING)

        if len(post_processing_queue) > 0:
            scheduled_status = RequestStatus.POSTPROCESSING
            return SchedulerOutput(
                scheduled_requests=convert_list_to_res_dict(post_processing_queue),
                status=scheduled_status,
            )

        prepare_processing_queue = self._get_all_reqs_by_status(RequestStatus.PREPARE)

        if len(prepare_processing_queue) > 0:
            scheduled_status = RequestStatus.PREPARE
            return SchedulerOutput(
                scheduled_requests=convert_list_to_res_dict(prepare_processing_queue),
                status=scheduled_status,
            )
        # Set slack
        for req in flattened_reqs:
            req.set_slack(self.model_name, False, self.predict_time)

        # Find the highest priority req
        now = time.time()
        index = 0
        # 修改arrival time到slack
        flattened_reqs.sort(key = lambda req: req.arrival_time, reverse=False)
        target_req = flattened_reqs[index]
        target_res = target_req.sampling_params.resolution
        target_status = target_req.status

        res_reqs_dict: Dict[int, Dict[int, Request]] = {}
        req_ids_to_abort: List[int] = []
        is_sliced = None
        patch_size = None
        if target_status == WorkerReqStatus.PREPARE:
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
        elif target_status == WorkerReqStatus.POSTPROCESSING:
            queue = self._get_all_reqs_by_status(target_status)
            queue.sort(key=lambda req: now - req.arrival_time, reverse=True)
            num_to_collect = max_num

            while num_to_collect > 0 and queue:
                req = queue.pop(0)
                res = req.sampling_params.resolution
                if res != target_res:
                    # We add only one resolution for postprocessing stage
                    continue
                elif res not in res_reqs_dict:
                    res_reqs_dict[res] = {req.request_id : req}
                else:
                    # We only add reqs that can satisfy SLO
                    if (self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])] 
                            / target_req.remain_time) < self.postprocessing_ratio:
                        res_reqs_dict[res][req.request_id] = req
                        self.predict_time = self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])]
                    else:
                        break
                num_to_collect -= 1
        elif target_status == WorkerReqStatus.DENOISING:
            running_reqs_list: List['Request'] = list()
            # Pick up running denoising reqs
            for req in flattened_reqs:
                if req.start_denoising and req.status == WorkerReqStatus.DENOISING:
                    res = req.sampling_params.resolution
                    # Add reqs that are currently running. They are destined to continue running.
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        res_reqs_dict[res][req.request_id] = req
                    running_reqs_list.append(req)
            if len(running_reqs_list) == 0:
                num_to_collect = max_num
            else:
                num_to_collect = 0
            while num_to_collect > 0:
                # Add new requests to start denoising
                if not target_req.start_denoising:
                    if len(running_reqs_list) > 0:
                        # Collect number of activated reqs by resolution.
                        res_candidate_dict: Dict[int, int] = {res : len(res_reqs_dict[res]) for res in res_reqs_dict}
                        for res in self.resolution_list:
                            if res not in res_candidate_dict:
                                res_candidate_dict[res] = 0

                        # Choose the resolution that can maximize throughput
                        # This is actually choosing the minimal resolution that has unfinifhsed reqs
                        best_tp_res = None
                        for res in self.resolution_list:
                            reqs_list = self.request_pool[res].get_all_unfinished_normal_reqs()
                            if len(reqs_list) != 0:
                                find_best_tp_res = False
                                for req in reqs_list:
                                    if req.status == WorkerReqStatus.DENOISING and not req.start_denoising:
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

                        # print(f"{task_distribute_original=}")
                        # print(f"{task_distribute_slo=}")
                        # print(f"{task_distribute_best_tp=}")
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

                        running_reqs_list.sort(key = lambda req: abs(req.slack), reverse=False)
                        # running_reqs_list.sort(key = lambda req: req.arrival_time, reverse=False)
                        # If we have low slack req, i.e., very urgent one, we don't add more reqs.
                        # TODO: A Hyper parameter here
                        # if running_reqs_list[0].slack < 0.1:
                        if False:
                            print(f"esymred: req {running_reqs_list[0].request_id} has slack {running_reqs_list[0].slack}. Very Urgent. Stop adding more reqs.")
                            break

                        # if target_req.slack < 0:
                        if False:
                            print(f"esymred: decide to abort request {target_req.request_id} with slack={target_req.slack}")
                            req_ids_to_abort.append(target_req.request_id)
                        else:
                            # 如果最紧急的请求依然不是特别紧急，则追求最大吞吐量，修改target_req为最小resolution的request
                            # if target_req.slack > 50:
                            # if True:
                            if False:
                                is_get_best_tp = True
                                for i in range(index, len(flattened_reqs)):
                                    req = flattened_reqs[i]
                                    if (req.sampling_params.resolution == best_tp_res 
                                        and req.status == WorkerReqStatus.DENOISING and not req.start_denoising):
                                        target_req = req
                                        target_res = best_tp_res
                                        print(f"esymred: req {target_req.request_id} not urgent with slack={target_req.slack}. "
                                              f"We use best_tp_res={target_res}")
                                        break
                            else:
                                print(f"esymred: req {target_req.request_id} is urgent with slack={target_req.slack}. "
                                        f"We use don't try for higher throughput.")
                                is_get_best_tp = False
                            # If some reqs are urgent, we prioritize satifying SLO.
                            if not is_get_best_tp:
                                if target_req_pred_time / (spend_time_original + self.predictor.get_latency(target_res) * target_req.remain_steps) < 0.95:
                                    if target_res not in res_reqs_dict:
                                        res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                    else:
                                        res_reqs_dict[target_res][target_req.request_id] = target_req
                                    target_req.start_denoising = True
                                    num_to_collect -= 1
                                    running_reqs_list.append(target_req)
                                    print(f"esymred: We add req {target_req.request_id} to satisfy SLO.")
                                    self.predict_time = predict_time_slo[0]
                                else:
                                    break
                            else:
                                # Add this req for best throughput is it won't significantly 
                                # TODO: A Hyper parameter here
                                if spend_time_best_tp / (spend_time_original + self.predictor.get_latency(target_res) * target_req.remain_steps) < 0.95:
                                    if target_res not in res_reqs_dict:
                                        res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                    else:
                                        res_reqs_dict[target_res][target_req.request_id] = target_req
                                    target_req.start_denoising = True
                                    running_reqs_list.append(target_req)
                                    num_to_collect -= 1
                                    print(f"esymred: We add req {target_req.request_id} to get higher throughput.")
                                    self.predict_time = predict_time_best_tp[0]
                                else:
                                    print(f"esymred: Adding req {target_req.request_id} cannot get higher throughput. Stop adding more req.")
                                    break
                    # end if len(running_reqs_list) > 0:
                    else:
                        # No currently activated reqs
                        # So add current req directly
                        # TODO: Why not check the slack value? Unconditionally add?
                        task_distribute = [[]]
                        for res in self.resolution_list:
                            if res == target_res:
                                task_distribute[0].append(1)
                            else:
                                task_distribute[0].append(0)
                        self.predict_time = self.predictor.predict(task_distribute)

                        target_req.update_predict_time(self.predict_time * target_req.remain_steps)
                        target_req.set_slack(self.model_name, True, self.predict_time) 

                        # if target_req.slack < 0:
                        if False:
                            print(f"esymred: decide to abort request {target_req.request_id} with slack={target_req.slack}")
                            req_ids_to_abort.append(target_req.request_id)
                        else:
                            if target_res not in res_reqs_dict:
                                res_reqs_dict[target_res] = {target_req.request_id : target_req}
                            else:
                                res_reqs_dict[target_res][target_req.request_id] = target_req
                            target_req.start_denoising = True
                            running_reqs_list.append(target_req)
                            num_to_collect -= 1
                # end if not target.start_denoising

                index += 1
                if index < len(flattened_reqs):
                    target_req = flattened_reqs[index]
                    target_res = target_req.sampling_params.resolution
                    target_status = target_req.status
                    # We don't add more denoising reqs that are preceded by POSTPROCESSING
                    # or PREPARE reqs, to prevent those reqs from expiration.
                    if target_status != WorkerReqStatus.DENOISING:
                        target_status = WorkerReqStatus.DENOISING
                        break
                else:
                    break
                if num_to_collect <= 0:
                    break
            # end while
            '''
            if len(res_reqs_dict) > 1:
                is_sliced = True
                patch_size = find_gcd(list(res_reqs_dict))
            elif len(res_reqs_dict) == 1:
                is_sliced = False
                patch_size = list(res_reqs_dict.keys())[0]
            '''
            is_sliced = True
            patch_size = 256
        # print(f"schedule time = {time.time() - start}")
        return SchedulerOutput(
            scheduled_requests=res_reqs_dict,
            status=target_status if len(res_reqs_dict) > 0 else WorkerReqStatus.EMPTY,
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
        
        return task_distribute_original, task_distribute_slo, task_distribute_best_tp
    
    
    def _get_spend_time(
        self,
        predict_time_original: List[float],
        predict_time_slo: List[float],
        predict_time_best_tp: List[float],
        target_req: 'Request',
        running_reqs_list: List['Request'],
    ) -> Tuple[float, float, float, float]:
        # print(f"{predict_time_original=}")
        # print(f"{predict_time_slo=}")
        # print(f"{predict_time_best_tp=}")
        remain_steps = running_reqs_list[0].remain_steps
        finish_tasks = 1
        spend_time = predict_time_slo[0] * remain_steps
        # 统计原激活队列与新加最大吞吐量的情况的预测执行时间
        spend_time_original = predict_time_original[0] * remain_steps
        spend_time_best_tp = predict_time_best_tp[0] * remain_steps
        running_reqs_list[0].update_predict_time(spend_time)
        # ! FIXME: Is self.predict correct at the first denoising round of first req?
        running_reqs_list[0].set_slack(self.model_name, True, self.predict_time)
        predict_index = 0
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

        # print(f"{spend_time=}")
        # print(f"{spend_time_original=}")
        # print(f"{spend_time_best_tp=}")
        # print(f"{target_req_pred_time=}")
        return (spend_time, spend_time_original, spend_time_best_tp, target_req_pred_time)