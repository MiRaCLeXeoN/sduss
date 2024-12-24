
import joblib
import torch.nn as nn
import sys, torch, os
import numpy as np
from sduss.utils import get_os_env
upsample_predictor = joblib.load(get_os_env("ESYMRED_UPSAMPLE_PATH", check_none=True))
downsample_predictor = joblib.load(get_os_env("ESYMRED_DOWNSAMPLE_PATH", check_none=True))

MAX = float(sys.maxsize)
class CacheManager:
    def __init__(self):
        super().__init__()
        self.table_index = []
        self.cache = dict()
        self.previous_mask = dict()
        self.mse_loss = nn.MSELoss(reduction='none')

    def save_and_get_block_states(self, new_indices, new_output, mask):
        # return new_output
        if mask.sum() == 0:
            output = torch.stack([self.cache[new_indices[index]] for index in range(len(new_indices))])
            self.cache = {new_indices[index] : output[index] for index in range(len(new_indices))}
            return output
        else:
            self.cache = {new_indices[index] : new_output[index] for index in range(len(new_indices))}
            return new_output
        
    def save_and_get_block_tupple(self, new_indices, new_output, mask, output_num):
        # return new_output
        if mask.sum() == 0:
            output = tuple([torch.stack([self.cache[new_indices[index]][i] for index in range(len(new_indices))]) for i in range(output_num)])
            # for i in range(output_num):
            #     state = torch.stack([self.cache[new_indices[index]][i] for index in range(len(new_indices))])
            #     output += (state,)
            # output = torch.stack([self.cache[new_indices[index]] for index in new_indices])
            self.cache = {new_indices[index]:[res[index] for res in output] for index in range(len(new_indices))}
            return output
        else:
            self.cache = {new_indices[index]:[res[index] for res in new_output] for index in range(len(new_indices))}
            return new_output

    def update_and_return(self, new_indices, new_output, mask, output_num = 0):
        # return new_output
        output = torch.empty((len(new_indices), *new_output.shape[1:]), device="cuda", dtype=torch.float16)
        
        if mask.sum() != len(new_indices):
            output[[new_indices[index] in self.cache and mask[index] == 0 for index in range(len(new_indices))]] = torch.stack([self.cache[new_indices[index]] for index in range(len(new_indices)) if new_indices[index] in self.cache and mask[index] == 0 ])
        if mask.sum() != 0:
            output[mask] = new_output
            self.cache = {new_indices[index] : output[index] for index in range(len(new_indices))}

        return output
    
    def get_mask(self, new_indices, new_input, total_blocks, timestep, is_upsample, res_tuple=None):
        
        # return np.array([1 for _ in range(new_input.shape[0])]) > 0.5
        common_keys = list(set(self.cache.keys()) & set(new_indices))
        if is_upsample:
            # mask[]
            C = torch.full((new_input.shape[0],len(res_tuple) + 1), MAX, device="cuda")
            if len(common_keys) != 0:
                A_tensors = torch.stack([self.cache[k][0] for k in common_keys])
                B_tensors = torch.stack([new_input[new_indices.index(k)] for k in common_keys])
                mse_values = self.mse_loss(A_tensors, B_tensors).mean(dim=(-1,-2,-3)).to(dtype=torch.float32)
                mse = [mse_values.unsqueeze(1)]
                # C[[k in self.cache for k in new_indices]][:, 0] = mse_values
                for i in range(len(res_tuple)):
                    A_tensors = torch.stack([self.cache[k][1][i] for k in common_keys])
                    B_tensors = torch.stack([res_tuple[i][new_indices.index(k)] for k in common_keys])
                    mse_values = self.mse_loss(A_tensors, B_tensors).mean(dim=(-1,-2,-3)).to(dtype=torch.float32)
                    mse.append(mse_values.unsqueeze(1))
                mse = torch.cat(mse, dim=1)
                C[[k in self.cache for k in new_indices]] = mse
            blocks = torch.full((new_input.shape[0], 1), int(total_blocks))
            timesteps = timestep.cpu()
            input_feature = torch.cat([blocks, timesteps.reshape(-1, 1), C.cpu()], dim=1).numpy()
            # input_feature = [[total_blocks, timestep[index], MAX] + [MAX for _ in res_tuple] if new_indices[index] not in self.cache else [total_blocks, timestep[index], self.mse_loss(new_input[index], self.cache[new_indices[index]][0]).mean(dim=(-1,-2,-3)).item()] + [self.mse_loss(res_tuple[x][index], self.cache[new_indices[index]][1][x]).mean(dim=(-1,-2,-3)).item() for x in range(len(res_tuple))] for index in range(len(new_indices))]

            self.previous_mask = {new_indices[index] : (0 if new_indices[index] not in self.cache 
            else self.previous_mask[new_indices[index]]) for index in range(len(new_indices)) }

            self.cache = {new_indices[index]:(new_input[index], [res[index] for res in res_tuple]) for index in range(len(new_indices))}
            # print(self.previous_mask)
            mask = upsample_predictor.predict(np.array(input_feature))
            
            mask[[self.previous_mask[new_indices[index]] == 4 for index in range(len(new_indices))]] = 1
            self.previous_mask = {new_indices[index]:(0 if mask[index] == 1 or self.previous_mask[new_indices[index]] == 4 else self.previous_mask[new_indices[index]] + 1) for index in range(len(new_indices))}
        else:
            C = torch.full((new_input.shape[0],), MAX, device="cuda")
            if len(common_keys) != 0:
                A_tensors = torch.stack([self.cache[k] for k in common_keys])
                B_tensors = torch.stack([new_input[new_indices.index(k)] for k in common_keys])
                mse_values = self.mse_loss(A_tensors, B_tensors).mean(dim=(-1,-2,-3)).to(dtype=torch.float32)
                C[[k in self.cache for k in new_indices]] = mse_values
            blocks = torch.full((new_input.shape[0], 1), int(total_blocks))
            timesteps = timestep.cpu()
            input_feature = torch.cat([blocks, timesteps.reshape(-1, 1), C.reshape(-1, 1).cpu()], dim=1).numpy()
            # input_feature = [[total_blocks, timestep[index], MAX] if new_indices[index] not in self.cache else [total_blocks, timestep[index], self.mse_loss(new_input[index], self.cache[new_indices[index]]).mean(dim=(-1,-2,-3)).item()] for index in range(len(new_indices))]
            # input_feature = [[total_blocks, timestep[index], MAX] for index in range(len(new_indices))]
            self.previous_mask = {new_indices[index] : (0 if new_indices[index] not in self.cache 
            else self.previous_mask[new_indices[index]]) for index in range(len(new_indices)) }
            mask = downsample_predictor.predict(np.array(input_feature))
            self.cache = {new_indices[index]:new_input[index] for index in range(len(new_indices))}

            # mask = [1 if mask[index] > 0.5 or (self.previous_mask[new_indices[index]] == 4) else 0 for index in range(len(new_indices))]
            mask[[self.previous_mask[new_indices[index]] == 4 for index in range(len(new_indices))]] = 1

            self.previous_mask = {new_indices[index]:(0 if mask[index] == 1 or self.previous_mask[new_indices[index]] == 4 else self.previous_mask[new_indices[index]] + 1) for index in range(len(new_indices))}
        mask = np.array(mask) > 0.5
        return mask
