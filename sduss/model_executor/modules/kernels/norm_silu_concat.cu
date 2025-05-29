#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/native/group_norm.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Dispatch.h>
#include <math.h>
// #include <ATen/native/cuda/group_norm_kernel.cu>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif
#define WARP_SIZE 32

template <typename T>
__device__ inline  T silu_tailer(T x) {
  // T ex = x + x * x * T(0.5) + x * x * x * T(0.166666) + x * x * x * x* T(0.041666) + T(1.0f);
  T ex = T(exp(-x));
  return x * (1 / (ex + T(1.0f)));
}

// gridDim = batch_size * num_groups_per_batch
// blockDim = 512
template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int N,
    T* X,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> rstd) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp =
      at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;

  const int i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = at::native::cuda_utils::WarpReduce(val, welford_op);
  } else {
    // There will be a warning if we declare a __shared__ WelfordType array.
    // https://github.com/pytorch/pytorch/pull/13967
    __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
        alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = at::native::cuda_utils::BlockReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = m2;
    // rstd[i] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}


// grid_dim = num_groups_per_batch * batch_size
// Only Norm and concat, no SiLU
template <typename T>
__global__ void NormSiluConcatCUDAKernel(
    torch::PackedTensorAccessor<T,4,torch::RestrictPtrTraits> X,
    int group, // channels per group
    int num_groups, // gorups per batch
    int H,
    int W,
    bool padding,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> mean,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> rstd,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> gamma,
    torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor<T,4,torch::RestrictPtrTraits> Y,
    // torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> latent_offset,
    // torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> patch_map, // have the same size with batch size, every elements corresponding the value of its latent idx
    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> padding_idx) {
  using T_ACC = at::acc_type<T, true>;
  const int thread_index = threadIdx.x;
  const int group_idx = blockIdx.x;
  const int group_idx_in_batch = group_idx % num_groups; 
  const int lane_idx = thread_index % WARP_SIZE;
  const int warp_id = thread_index / WARP_SIZE;
  const int channels_per_group = blockDim.x / WARP_SIZE; // channels one thread_block can process once
  
  extern __shared__ int sh[];
  // T* full_mean_value = (T*)sh;
  // T* full_rstd_value = (T*)&full_mean_value[1];
  // __shared__ T full_mean_value[1];
  // __shared__ T full_rstd_value[1];
  // __shared__ T result_value[group][H][W];
  T* result_value = (T*)&sh;
  const int batch_idx = group_idx / num_groups; 
  // const int latent_idx = patch_map[batch_idx];
  // const int first_batch_idx = latent_offset[latent_idx - 1];
  // const int total_patches = latent_offset[latent_idx] - latent_offset[latent_idx - 1];
  // const int idx = first_batch_idx * num_groups + group_idx_in_batch; // corresponding group idx of the first patch 
  T full_mean = mean[group_idx];
  T full_rstd = rstd[group_idx];
  // if (thread_index == 0) {
    // for (int32_t j = 1; j < total_patches; j++) {
    //   full_mean += mean[idx + num_groups * j];
    //   full_rstd += rstd[idx + num_groups * j];
    // }
    // full_mean_value[0] = full_mean;
    // full_rstd_value[0] = full_rstd;
  //   full_mean_value[0] = T(full_mean / (total_patches));
  //   full_rstd_value[0] = c10::cuda::compat::rsqrt(T(full_rstd / (total_patches)) + static_cast<T_ACC>(eps));
  // }
  // __syncwarp();
  // full_mean = full_mean_value[0];
  // full_rstd = full_rstd_value[0];
  // new_mean[group_idx] = full_mean;
  // new_rstd[group_idx] = full_rstd;
  // const int base_index = H * W * group * group_idx;
  // const int offset = H * W * warp_id;
  // const int result_base_index = (H + 2) * (W + 2) * group * group_idx;
  // const int result_offset = (H + 2) * (W + 2) * warp_id;
  // const int new_HxW = (H + 2) * (W + 2);
  // const int stride = channels_per_group * HxW;
  // const int new_stride = channels_per_group * new_HxW;
  const int skip_channels = group_idx_in_batch * group;
  const int padding_batch_idx_top = padding_idx[batch_idx * 4];
  const int padding_batch_idx_left = padding_idx[batch_idx * 4 + 1];
  const int padding_batch_idx_bottom = padding_idx[batch_idx * 4 + 2];
  const int padding_batch_idx_right = padding_idx[batch_idx * 4 + 3];
  for (int32_t j = 0; j < (group / channels_per_group); j++) {
    int channel_in_group = j * channels_per_group + warp_id;
    for (int32_t k = lane_idx; k < H*W; k += WARP_SIZE) {
      
      int rows = k / W;
      int cols = k % W;
      T scale = full_rstd * gamma[skip_channels + channel_in_group];
      T beta_data = -scale * full_mean + beta[skip_channels + channel_in_group];
      // T scale = full_rstd;
      // T beta_data = -scale * full_mean;
      // T x = X[base_index + j * stride + offset + k] * scale + beta_data;
      // T x = (X[batch_idx][skip_channels + channel_in_group][rows][cols]);
      T x = (X[batch_idx][skip_channels + channel_in_group][rows][cols] * scale + beta_data);
      if (padding) {
        if (padding_batch_idx_top != -1 && rows == 0) {
          result_value[channel_in_group * W + cols] = x;
        } else if (padding_batch_idx_bottom != -1 && rows == H - 1) {
          result_value[group * W + channel_in_group * W + cols] = x;
        }
        if (padding_batch_idx_left != -1 && cols == 0) {
          result_value[group * W * 2 + channel_in_group * H + rows] = x;
        } else if (padding_batch_idx_right != -1 && cols == W - 1) {
          result_value[group * W * 2 + group * H + channel_in_group * H + rows] = x;
        }
        Y[batch_idx][skip_channels + channel_in_group][rows + 1][cols + 1] = x;
      } else {
        Y[batch_idx][skip_channels + channel_in_group][rows][cols] = x;
      }
      
      // result_value[channel_in_group * HxW + k] = silu_tailer(x);
      // result_value[j * stride + offset + k] = (x * ex) / (ex + 1);
    }
  }
  if(padding) {
    __syncthreads();
    for (int32_t k = thread_index; k < group*W; k += blockDim.x) {
      const int channel_in_group_idx = k / W;
      const int id_in_row = k % W;
      if (padding_batch_idx_top != -1) {
        // Y[eles_per_batch * padding_batch_idx_top + offset_in_batch + channel_in_group_idx * new_HxW + id_in_row + 1] 
        //   = result_value[channel_in_group_idx * HxW + id_in_row];
        Y[padding_batch_idx_top][skip_channels + channel_in_group_idx][H + 1][id_in_row + 1]
          = result_value[channel_in_group_idx * W + id_in_row];
      }
      if (padding_batch_idx_bottom != -1) {
        // Y[eles_per_batch * padding_batch_idx_bottom + offset_in_batch + channel_in_group_idx * new_HxW +  + ((H + 1) * (W + 2)) + id_in_row + 1]
        //   = result_value[channel_in_group_idx * HxW + (H - 1) * W + id_in_row];
        Y[padding_batch_idx_bottom][skip_channels + channel_in_group_idx][0][id_in_row + 1]
          = result_value[group * W + channel_in_group_idx * W + id_in_row];
      }
    }
    for (int32_t k = thread_index; k < group*H; k += blockDim.x) {
      const int channel_in_group_idx = k / H;
      const int id_in_col = k % H;
      if (padding_batch_idx_left != -1) {
        // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 1) * (W + 2)] 
        //   = result_value[channel_in_group_idx * HxW + id_in_col * W];
        Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][id_in_col + 1][W+1]
          = result_value[group * W * 2 + channel_in_group_idx * H + id_in_col];
        if (id_in_col == 0) {
          // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW] 
          //   = result_value[channel_in_group_idx * HxW];
          Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][0][W+1] 
            = result_value[group * W * 2 + channel_in_group_idx * H];
        }
        if (id_in_col == H - 1) {
          // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 2) * (W + 2)] 
          //   = result_value[channel_in_group_idx * HxW + id_in_col * W];
          Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][H+1][W+1] 
            = result_value[group * W * 2 + channel_in_group_idx * H + id_in_col];
        }
      }
      if (padding_batch_idx_right != -1) {
        // Y[(H + 2) * (W + 2) * C * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 1) * (W + 2) + (W + 1)] 
        //   = result_value[channel_in_group_idx * HxW + id_in_col * W + (W - 1)];
        Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][id_in_col + 1][0]
          = result_value[group * W * 2 + group * H + channel_in_group_idx * H + id_in_col];
        if (id_in_col == 0) {
          // Y[eles_per_batch * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (W + 1)] 
          //   = result_value[channel_in_group_idx * HxW + (W - 1)];
          Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][0][0]
            = result_value[group * W * 2 + group * H + channel_in_group_idx * H];
        }
        if (id_in_col == H - 1) {
          // Y[eles_per_batch * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 2) * (W + 2) + (W + 1)] 
          //   = result_value[channel_in_group_idx * HxW + id_in_col * W + (W - 1)];
          Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][H + 1][0]
            = result_value[group * W * 2 + group * H + channel_in_group_idx * H + id_in_col];
        }
      }
    }

  }
}

// Filling up
template <typename T>
__global__ void MockNormSiluConcatCUDAKernel(
    torch::PackedTensorAccessor<T,4,torch::RestrictPtrTraits> X,
    int group, // channels per group
    int num_groups, // gorups per batch
    int H,
    int W,
    torch::PackedTensorAccessor<T,4,torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> padding_idx) {
  using T_ACC = at::acc_type<T, true>;
  const int thread_index = threadIdx.x;
  const int group_idx = blockIdx.x;
  const int group_idx_in_batch = group_idx % num_groups; 
  const int lane_idx = thread_index % WARP_SIZE;
  const int warp_id = thread_index / WARP_SIZE;
  const int channels_per_group = blockDim.x / WARP_SIZE; // channels one thread_block can process once
  
  extern __shared__ int sh[];

  T* result_value = (T*)&sh;
  const int batch_idx = group_idx / num_groups; 

  // T full_mean = mean[group_idx];
  // T full_rstd = rstd[group_idx];

  const int skip_channels = group_idx_in_batch * group;
  const int padding_batch_idx_top = padding_idx[batch_idx * 4];
  const int padding_batch_idx_left = padding_idx[batch_idx * 4 + 1];
  const int padding_batch_idx_bottom = padding_idx[batch_idx * 4 + 2];
  const int padding_batch_idx_right = padding_idx[batch_idx * 4 + 3];
  for (int32_t j = 0; j < (group / channels_per_group); j++) {
    int channel_in_group = j * channels_per_group + warp_id;
    for (int32_t k = lane_idx; k < H*W; k += WARP_SIZE) {
      
      int rows = k / W;
      int cols = k % W;

      T x = (X[batch_idx][skip_channels + channel_in_group][rows][cols]);
      if (padding_batch_idx_top != -1 && rows == 0) {
        result_value[channel_in_group * W + cols] = x;
      } else if (padding_batch_idx_bottom != -1 && rows == H - 1) {
        result_value[group * W + channel_in_group * W + cols] = x;
      }
      if (padding_batch_idx_left != -1 && cols == 0) {
        result_value[group * W * 2 + channel_in_group * H + rows] = x;
      } else if (padding_batch_idx_right != -1 && cols == W - 1) {
        result_value[group * W * 2 + group * H + channel_in_group * H + rows] = x;
      }
      Y[batch_idx][skip_channels + channel_in_group][rows + 1][cols + 1] = x;
      // result_value[channel_in_group * HxW + k] = silu_tailer(x);
      // result_value[j * stride + offset + k] = (x * ex) / (ex + 1);
    }
  }

  __syncthreads();
  for (int32_t k = thread_index; k < group*W; k += blockDim.x) {
    const int channel_in_group_idx = k / W;
    const int id_in_row = k % W;
    if (padding_batch_idx_top != -1) {
      // Y[eles_per_batch * padding_batch_idx_top + offset_in_batch + channel_in_group_idx * new_HxW + id_in_row + 1] 
      //   = result_value[channel_in_group_idx * HxW + id_in_row];
      Y[padding_batch_idx_top][skip_channels + channel_in_group_idx][H + 1][id_in_row + 1]
        = result_value[channel_in_group_idx * W + id_in_row];
    }
    if (padding_batch_idx_bottom != -1) {
      // Y[eles_per_batch * padding_batch_idx_bottom + offset_in_batch + channel_in_group_idx * new_HxW +  + ((H + 1) * (W + 2)) + id_in_row + 1]
      //   = result_value[channel_in_group_idx * HxW + (H - 1) * W + id_in_row];
      Y[padding_batch_idx_bottom][skip_channels + channel_in_group_idx][0][id_in_row + 1]
        = result_value[group * W + channel_in_group_idx * W + id_in_row];
    }
  }
  for (int32_t k = thread_index; k < group*H; k += blockDim.x) {
    const int channel_in_group_idx = k / H;
    const int id_in_col = k % H;
    if (padding_batch_idx_left != -1) {
      // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 1) * (W + 2)] 
      //   = result_value[channel_in_group_idx * HxW + id_in_col * W];
      Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][id_in_col + 1][W+1]
        = result_value[group * W * 2 + channel_in_group_idx * H + id_in_col];
      if (id_in_col == 0) {
        // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW] 
        //   = result_value[channel_in_group_idx * HxW];
        Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][0][W+1] 
          = result_value[group * W * 2 + channel_in_group_idx * H];
      }
      if (id_in_col == H - 1) {
        // Y[eles_per_batch * padding_batch_idx_left + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 2) * (W + 2)] 
        //   = result_value[channel_in_group_idx * HxW + id_in_col * W];
        Y[padding_batch_idx_left][skip_channels + channel_in_group_idx][H+1][W+1] 
          = result_value[group * W * 2 + channel_in_group_idx * H + id_in_col];
      }
    }
    if (padding_batch_idx_right != -1) {
      // Y[(H + 2) * (W + 2) * C * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 1) * (W + 2) + (W + 1)] 
      //   = result_value[channel_in_group_idx * HxW + id_in_col * W + (W - 1)];
      Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][id_in_col + 1][0]
        = result_value[group * W * 2 + group * H + channel_in_group_idx * H + id_in_col];
      if (id_in_col == 0) {
        // Y[eles_per_batch * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (W + 1)] 
        //   = result_value[channel_in_group_idx * HxW + (W - 1)];
        Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][0][0]
          = result_value[group * W * 2 + group * H + channel_in_group_idx * H];
      }
      if (id_in_col == H - 1) {
        // Y[eles_per_batch * padding_batch_idx_right + offset_in_batch + channel_in_group_idx * new_HxW + (id_in_col + 2) * (W + 2) + (W + 1)] 
        //   = result_value[channel_in_group_idx * HxW + id_in_col * W + (W - 1)];
        Y[padding_batch_idx_right][skip_channels + channel_in_group_idx][H + 1][0]
          = result_value[group * W * 2 + group * H + channel_in_group_idx * H + id_in_col];
      }
    }
  }
}

template <typename T>
__global__ void GetFullMeanAndRstd(
  int num_groups,
  T eps,
  torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> mean,
  torch::PackedTensorAccessor<T,1,torch::RestrictPtrTraits> rstd,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> latent_offset,
  torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits> patch_map
) {
  using T_ACC = at::acc_type<T, true>;
  const int thread_index = threadIdx.x;
  const int batch_idx = blockIdx.x;
  const int latent_idx = patch_map[batch_idx];
  const int first_batch_idx = latent_offset[latent_idx - 1];
  const int total_patches = latent_offset[latent_idx] - latent_offset[latent_idx - 1];
  const int idx = first_batch_idx * num_groups + thread_index;
  if (thread_index < num_groups) {
    T full_mean = mean[idx];
    T full_rstd = rstd[idx];
    for (int32_t j = 1; j < total_patches; j++) {
      full_mean += mean[idx + num_groups * j];
      full_rstd += rstd[idx + num_groups * j];
    }
    mean[batch_idx * num_groups + thread_index] = T(full_mean / (total_patches));
    rstd[batch_idx * num_groups + thread_index] = c10::cuda::compat::rsqrt(T(full_rstd / (total_patches)) + static_cast<T_ACC>(eps));
  }
}

void GetMeanAndRstd(
    const torch::Tensor& X,
    int N,
    int C,
    int H,
    int W,
    int group,
    double eps,
    torch::Tensor& mean,
    torch::Tensor& rstd,
    torch::Tensor& latent_offset,
    torch::Tensor& patch_map
) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int num_threads = 512;
  const int HxW = H * W;
  const int G = group;
  const int D = C / G;
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
      X.scalar_type(),
      "RowwiseMoments",
      ([&]  __global__ {
        RowwiseMomentsCUDAKernel<scalar_t><<<N * D, num_threads, 0, cuda_stream>>>(
            G * HxW, 
            // X.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            X.data_ptr<scalar_t>(),
            mean.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            rstd.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>());
      }));
      cudaDeviceSynchronize();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mean.scalar_type(),
      "GetFullMeanAndRstdImpl",
      ([&] __global__  {
        GetFullMeanAndRstd<scalar_t><<<N, 32, 0, cuda_stream>>>(
            D, static_cast<scalar_t>(eps), 
            mean.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(), 
            rstd.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(), 
            latent_offset.packed_accessor<int,1,torch::RestrictPtrTraits>(), 
            patch_map.packed_accessor<int,1,torch::RestrictPtrTraits>());
      }));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "RowwiseMomentsCUDAKernel kernel error: " << cudaGetErrorString(error) << std::endl;
  }
}

void FuseGroupNormKernelImplInternal(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int N,
    int C,
    int H,
    int W,
    int group,
    bool padding,
    torch::Tensor& Y,
    torch::Tensor& mean,
    torch::Tensor& rstd,
    
    torch::Tensor& padding_idx) {
  // using T_ACC = acc_type<T, true>;
    const int HxW = H * W;
    const int G = group;
    const int D = C / G;
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
    const int num_threads = 512;
    const int gridDim = D * N;
    const int blockDim = 320;

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "FuseGroupNormKernelImpl",
      ([&] __global__  {
        const int shared_mem = group * W * sizeof(scalar_t) * 2 + group * H * sizeof(scalar_t) * 2;
        NormSiluConcatCUDAKernel<scalar_t><<<gridDim, blockDim, shared_mem, cuda_stream>>>(
            X.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(), 
            G, D, H, W, padding,
            mean.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(), 
            rstd.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(), 
            gamma.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            beta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(), 
            Y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(), 
            // latent_offset.packed_accessor<int,1,torch::RestrictPtrTraits>(), 
            // patch_map.packed_accessor<int,1,torch::RestrictPtrTraits>(), 
            padding_idx.packed_accessor<int,1,torch::RestrictPtrTraits>());
      }));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "NormSiluConcatCUDAKernel kernel error: " << cudaGetErrorString(error) << std::endl;
  }
}

void MockFuseGroupNormKernelImplInternal(
    const torch::Tensor& X,
    int N,
    int C,
    int H,
    int W,
    int group,
    torch::Tensor& Y,
    torch::Tensor& padding_idx) {
  const int HxW = H * W;
  const int G = group;
  const int D = C / G;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int num_threads = 512;
  const int gridDim = D * N;
  const int blockDim = 320;
      AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "MockFuseGroupNormKernelImpl",
      ([&] __global__  {
        const int shared_mem = group * W * sizeof(scalar_t) * 2 + group * H * sizeof(scalar_t) * 2;
        MockNormSiluConcatCUDAKernel<scalar_t><<<gridDim, blockDim, shared_mem, cuda_stream>>>(
            X.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(), 
            G, D, H, W, 
            Y.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(), 
            padding_idx.packed_accessor<int,1,torch::RestrictPtrTraits>());
      }));
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "NormSiluConcatCUDAKernel kernel error: " << cudaGetErrorString(error) << std::endl;
  }
}