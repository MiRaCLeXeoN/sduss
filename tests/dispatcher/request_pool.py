from sduss.dispatcher.request_pool import RequestPool, Request
from sduss.model_executor.sampling_params import BaseSamplingParams

pool = RequestPool()

reqs = [Request(i, BaseSamplingParams(resolution=512)) for i in range(10)] 
for i, req in enumerate(reqs):
    req.dp_rank = i % 4

pool.add_requests(reqs)

print(pool.get_pixels_all_dp_rank())