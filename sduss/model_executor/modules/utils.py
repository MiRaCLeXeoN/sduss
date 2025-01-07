import torch
import math

def split_sample(samples, patch_size, input_indices):
        latent_offset = list()
        patch_map = list()
        latent_offset.append(0)
        resolution_offset = list()
        resolution_offset.append(0)
        padding_idx = list()
        new_sample = list()
        indices = list()
        for resolution, res_sample in samples.items():
            resolution = int(resolution)
            patch_on_height = (resolution // patch_size)
            patch_on_width = (resolution // patch_size)
            latent_patch_size = int(patch_size // 8)
            if res_sample is None or res_sample.shape[0] == 0:
                continue
            index = 0
            for sample in res_sample:
                latent_offset.append(latent_offset[-1] + patch_on_width ** 2)
                sample = torch.nn.functional.pad(sample, (1, 1, 1, 1), "constant", 0).unsqueeze(0)
                
                for h in range((patch_on_height)):
                    for w in range((patch_on_width)):
                        paddings = torch.empty(4, device=sample.device, dtype=torch.int32)
                        # paddings = [None] * 4
                        if (patch_on_height) == 1:
                            paddings[0] = -1
                            paddings[1] = -1
                            paddings[2] = -1
                            paddings[3] = -1
                            new_sample.append(sample[:, :, h * latent_patch_size : (h + 1) * latent_patch_size + 2, w * latent_patch_size : (w + 1) * latent_patch_size + 2])
                            patch_map.append(len(latent_offset)-1)
                            padding_idx.append(paddings)
                            indices.append(input_indices[str(resolution)][index] + f"-{h}-{w}")
                            continue
                        if w == 0:
                            paddings[1] = -1
                            paddings[3] = len(new_sample) + 1
                        elif w == (patch_on_width) - 1:
                            paddings[1] = len(new_sample) - 1
                            paddings[3] = -1
                        else:
                            paddings[1] = len(new_sample) - 1
                            paddings[3] = len(new_sample) + 1
                        if h == 0:
                            paddings[0] = -1
                            paddings[2] = len(new_sample) + (patch_on_height)
                        elif h == (patch_on_height) - 1:
                            paddings[0] = len(new_sample) - (patch_on_height)
                            paddings[2] = -1
                        else:
                            paddings[0] = len(new_sample) - (patch_on_height)
                            paddings[2] = len(new_sample) + (patch_on_height)
                        new_sample.append(sample[:, :, h * latent_patch_size : (h + 1) * latent_patch_size + 2, w * latent_patch_size : (w + 1) * latent_patch_size + 2])
                        patch_map.append(len(latent_offset)-1)
                        padding_idx.append(paddings)
                        # print(input_indices)
                        # print(resolution)
                        indices.append(input_indices[str(resolution)][index] + f"-{h}-{w}")
                index += 1
            if res_sample.shape[0] != 0:
                resolution_offset.append(len(latent_offset)-1)
        # padding_idx = [left_idx, top_idx, right_idx, bottom_idx]
        padding_idx = {
            "cuda": torch.cat(padding_idx, dim=0).cuda(),
            "cpu": padding_idx
        }
        latent_offset = {
            "cuda": torch.tensor(latent_offset, device="cuda", dtype=torch.int32),
            "cpu": latent_offset
        }
        resolution_offset = {
            "cuda": torch.tensor(resolution_offset, device="cuda", dtype=torch.int32),
            "cpu": resolution_offset
        }
        patch_map = {
            "cuda": torch.tensor(patch_map, device="cuda", dtype=torch.int32),
            "cpu": patch_map
        }
        return indices, padding_idx, latent_offset, resolution_offset, torch.cat(new_sample, dim=0), patch_map


def split_sample_sd3(samples, patch_size, input_indices):
        latent_offset = list()
        latent_offset.append(0)
        resolution_offset = list()
        resolution_offset.append(0)
        new_sample = list()
        indices = list()
        encoder_indices = list()
        for resolution, res_sample in samples.items():
            resolution = int(resolution)
            patch_on_height = (resolution // patch_size)
            patch_on_width = (resolution // patch_size)
            latent_patch_size = int(patch_size // 8)
            if res_sample is None or res_sample.shape[0] == 0:
                continue
            index = 0
            for sample in res_sample:
                latent_offset.append(latent_offset[-1] + patch_on_width ** 2)
                encoder_indices.append(input_indices[str(resolution)][index])
                for h in range((patch_on_height * patch_on_width)):
                    if (patch_on_height) == 1:
                        indices.append(input_indices[str(resolution)][index] + f"-{h}")
                        continue
                    indices.append(input_indices[str(resolution)][index] + f"-{h}")
                new_sample.extend(sample.chunk(patch_on_height * patch_on_width, dim=0))
                index += 1

            if res_sample.shape[0] != 0:
                resolution_offset.append(len(latent_offset)-1)
        # padding_idx = [left_idx, top_idx, right_idx, bottom_idx]
        latent_offset = {
            "cpu": latent_offset
        }
        resolution_offset = {
            "cpu": resolution_offset
        }
        return indices, encoder_indices, latent_offset, resolution_offset, torch.stack(new_sample)

def concat_sample(patch_size, new_sample, latent_offset):
        samples = dict()
        for index in range(len(latent_offset) - 1):
            patches_per_image = latent_offset[index + 1] - latent_offset[index]
            patch_on_height = int(math.sqrt(patches_per_image))
            image_size = patch_on_height * patch_size
            image_arr = new_sample[latent_offset[index] : latent_offset[index + 1]].view(1, -1, new_sample.shape[-1])
            if str(image_size) not in samples:
                samples[str(image_size)] = list()
            samples[str(image_size)].append(image_arr)
        for key in samples:
            samples[key] = torch.cat(samples[key], dim=0)
        return samples

