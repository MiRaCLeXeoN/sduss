import time

from typing import List, TYPE_CHECKING, Dict
import multiprocessing
from sduss.scheduler.wrappers import ResolutionRequestQueue
import heapq

from .policy import Policy
from ..wrappers import SchedulerOutput, RequestStatus
from ..utils import find_gcd
import joblib
import os

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
    """First Come First Serve.
    
    FCFS always selects the oldest requests.

    Support mixed precision.
    """

    def __init__(self, request_pool: Dict[int, ResolutionRequestQueue], model_path: str, postprocessing_exec_time_dir: str) -> None:
        super().__init__(request_pool)
        self.predictor = Predictor(model_path)
        if "sd1.5" in model_path:
            self.model_name = "sd1.5"
            self.resolution_list = [256, 512, 768]
        else:
            self.model_name = "sdxl"
            self.resolution_list = [512, 768, 1024]
        self.postprocessing_exec_time = {
            "sd1.5": {},
            "sdxl": {}
        }
        for resolution in self.resolution_list:
            path = os.path.join(postprocessing_exec_time_dir, f"sm_util_{self.model_name}_{resolution}.csv")
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
        # self.waiting_heap = []
        # self.active_heap = []
        self.predict_time = 0

    def _flatten_all_reqs(self) -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_unfinished_reqs())
        return reqs
    
    
    def _get_all_reqs_by_status(self, status: "RequestStatus") -> List['Request']:
        reqs = []
        for resolution_queue in self.request_pool.values():
            reqs.extend(resolution_queue.get_all_reqs_by_status(status))
        return reqs
    
    
    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Schedule requests for next iteration.

        FCFS features
            Supports:
                1. batch reqs of different timesteps
            Don't supports:
                2. mixed-precision shceduling

        Args:
            max_num (int): _description_

        Returns:
            List[Request]: _description_
        """
        # for index in range(len(self.waiting_heap)):
        #     self.waiting_heap[index].get_slack(False)
        # heapq.heapify(self.waiting_heap)
        flattened_reqs = self._flatten_all_reqs()

        for seq in flattened_reqs:
            seq.get_slack(self.model_name, False, self.predict_time)

        # Find the oldest request
        now = time.time()
        index = 0
        flattened_reqs.sort(key = lambda req: req.slack, reverse=False)
        target_req = flattened_reqs[index]
        target_res = target_req.sampling_params.resolution
        target_status = target_req.status
        res_reqs_dict: Dict[int, Dict[int, Request]] = {}
        # 处理prepare阶段与postprocessing阶段
        if target_status != RequestStatus.DENOISING:
            queue = self._get_all_reqs_by_status(target_status)
            queue.sort(key=lambda req: now - req.arrival_time, reverse=True)
            num_to_collect = max_num

            while num_to_collect > 0 and queue:
                req = queue.pop(0)
                res = req.sampling_params.resolution
                if target_status == RequestStatus.PREPARE:
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        res_reqs_dict[res][req.request_id] = req
                elif target_status == RequestStatus.POSTPROCESSING:
                    # postprocessing阶段不支持混合精度，必须得相同精度
                    if res != target_res:
                        continue
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        # 添加新的请求数量不能超过target请求的ddl
                        if self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])] / target_req.remain_time < 0.9:
                            res_reqs_dict[res][req.request_id] = req
                            self.predict_time = self.postprocessing_exec_time[self.model_name][str(res)][len(res_reqs_dict[res])]
                        else:
                            break
                num_to_collect -= 1

            return SchedulerOutput(
                scheduled_requests=res_reqs_dict,
                status=target_status,
                is_sliced=is_sliced,
                patch_size=patch_size,
            )
        else:
            # running_reqs_by_res = dict()
            running_reqs_list = list()
            # 将已经在运行中的请求挑选出来
            for req in flattened_reqs:
                if req.start_denoising and req.status == RequestStatus.DENOISING:
                    res = req.sampling_params.resolution
                    if res not in res_reqs_dict:
                        res_reqs_dict[res] = {req.request_id : req}
                    else:
                        res_reqs_dict[res][req.request_id] = req
                    # running_reqs_by_res[res].append(req)
                    running_reqs_list.append(req)
            
            while True:
                # 如果选取target请求已经被调度，则选择下一个
                if not target_req.start_denoising:
                
                    if len(running_reqs_list) > 0:
                        # for res in self.resolution_list:
                        #     # if res == target_res:
                        #     #     resolution_candidate_list[res] = (len(res_reqs_dict[res]) + 1)
                        #     # else:
                        #         resolution_candidate_list[res] = (len(res_reqs_dict[res]))
                        resolution_candidate_list = {x:len(res_reqs_dict[x]) for x in self.resolution_list}
                        # 选择可以最大化吞吐量的分辨率
                        best_tp_res = None
                        for res in self.resolution_list:
                            if len(self.request_pool[res].get_all_unfinished_reqs()) != 0:
                                find_best_tp_res = False
                                for req in self.request_pool[res].get_all_unfinished_reqs():
                                    if req.status == RequestStatus.DENOISING and not req.start_denoising:
                                        best_tp_res = res
                                        find_best_tp_res = True
                                        break
                                if find_best_tp_res:
                                    break
                        # 三种情况：当前任务分布、如果增加最紧急任务情况、如果增加最大化吞吐量任务情况
                        task_distribute_original = list()
                        task_distribute_original.append([resolution_candidate_list[res] for res in self.resolution_list])
                        task_distribute_slo = list()
                        task_distribute_slo.append([resolution_candidate_list[res] if res != target_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                        task_distribute_best_tp = list()
                        task_distribute_best_tp.append([resolution_candidate_list[res] if res != best_tp_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                        
                        running_reqs_list.sort(key = lambda req: req.remain_steps, reverse=False)
                        # 遍历待执行任务列表，计算每个任务在三种情况下的理论执行时间
                        finish_tasks = 1
                        remain_steps = running_reqs_list[0].remain_steps
                        resolution_candidate_list[running_reqs_list[0].sampling_params.resolution] -= 1
                        while finish_tasks < len(running_reqs_list):
                            req = running_reqs_list[finish_tasks]
                            # 如果剩余的任务的remain_step与之前不同，说明在req.remain_steps - remain_steps步之内，
                            # 执行的任务分布没有发生变化，直到这个任务也结束执行denoising
                            if req.remain_steps != remain_steps:
                                task_distribute_original.append([resolution_candidate_list[res] for res in self.resolution_list])
                                task_distribute_slo.append([resolution_candidate_list[res] if res != target_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                                task_distribute_best_tp.append([resolution_candidate_list[res] if res != best_tp_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                            resolution_candidate_list[req.sampling_params.resolution] -= 1
                            remain_steps = req.remain_steps
                            finish_tasks += 1
                        # task_distribute.append([resolution_candidate_list[res] for res in self.resolution_list])
                        # 如果激活队列中也存在还未执行的任务，则无需额外处理，否则，需要计算单独执行待添加的任务需要执行的时间
                        if remain_steps != target_req.remain_steps:
                            task_distribute_slo.append([resolution_candidate_list[res] if res != target_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                            task_distribute_best_tp.append([resolution_candidate_list[res] if res != best_tp_res else (resolution_candidate_list[res] + 1) for res in self.resolution_list])
                        
                        predict_time_original = self.predictor.predict(task_distribute_original)
                        predict_time_slo = self.predictor.predict(task_distribute_slo)
                        predict_time_best_tp = self.predictor.predict(task_distribute_best_tp)
                        remain_steps = running_reqs_list[0].remain_steps
                        finish_tasks = 1
                        spend_time = predict_time_slo[0] * remain_steps
                        # 统计原激活队列与新加最大吞吐量的情况的预测执行时间
                        spend_time_original = predict_time_original[0] * remain_steps
                        spend_time_best_tp = predict_time_best_tp[0] * remain_steps
                        running_reqs_list[0].update_predict_time(spend_time)
                        running_reqs_list[0].get_slack(self.model_name, True, self.predict_time)
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
                            req.get_slack(self.model_name, True, self.predict_time)
                            remain_steps = req.remain_steps
                            finish_tasks += 1
                        target_req_pred_time = (target_req.remain_steps - remain_steps) * predict_time_slo[-1] + spend_time
                        spend_time_best_tp += (target_req.remain_steps - remain_steps) * predict_time_best_tp[-1]
                        target_req.update_predict_time(target_req_pred_time)
                        target_req.get_slack(self.model_name, True, self.predict_time) 
                        # 查看激活队列中的请求，是否存在slack过低的
                        running_reqs_list.sort(key = lambda req: req.slack, reverse=False)
                        if running_reqs_list[0].slack < 0.1:
                            break
                        if target_req.slack < 0:
                            target_req.status = RequestStatus.FINISHED_ABORTED
                        else:
                            # 如果最紧急的请求依然不是特别紧急，则追求最大吞吐量，修改target_req为最小resolution的request
                            if target_req.slack > 1:
                                is_get_best_tp = True
                                for i in range(index, len(flattened_reqs)):
                                    req = flattened_reqs[i]
                                    if req.sampling_params.resolution == best_tp_res and req.status == RequestStatus.DENOISING and not req.start_denoising:
                                        target_req = req
                                        target_res = best_tp_res
                                        break
                            else:
                                is_get_best_tp = False
                            if not is_get_best_tp:
                                if target_res not in res_reqs_dict:
                                    res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                else:
                                    res_reqs_dict[target_res][target_req.request_id] = target_req
                                target_req.start_denoising = True
                                running_reqs_list.append(target_req)
                                self.predict_time = predict_time_slo[0]
                            else:
                                if spend_time_best_tp / (spend_time_original + self.predictor.predict([[1 if res == target_res else 0 for res in self.resolution_list]])[0]) < 0.95:
                                    if target_res not in res_reqs_dict:
                                        res_reqs_dict[target_res] = {target_req.request_id : target_req}
                                    else:
                                        res_reqs_dict[target_res][target_req.request_id] = target_req
                                    running_reqs_list.append(target_req)
                                    target_req.start_denoising = True
                                    self.predict_time = predict_time_best_tp[0]
                                else:
                                    break
                    else:
                        # 如果当前没有任何任务在激活队列中，则直接将当前任务加入激活队列
                        resolution_candidate_list = list()
                        for res in self.resolution_list:
                            if res == target_res:
                                resolution_candidate_list.append(1)
                            else:
                                resolution_candidate_list.append(0)
                        if target_res not in res_reqs_dict:
                            res_reqs_dict[target_res] = {target_req.request_id : target_req}
                        else:
                            res_reqs_dict[target_res][target_req.request_id] = target_req
                        target_req.start_denoising = True
                        running_reqs_list.append(target_req)
                        self.predict_time = self.predictor.predict(resolution_candidate_list)
                index += 1
                if index < len(flattened_reqs):
                    target_req = flattened_reqs[index]
                    target_res = target_req.sampling_params.resolution
                    target_status = target_req.status
                    # 如果接下来最紧急请求是postprocessing请求，则不会继续添加，防止postprocessing超时
                    if target_status != RequestStatus.DENOISING:
                        target_status = RequestStatus.DENOISING
                        break
                else:
                    break
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