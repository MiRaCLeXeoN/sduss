#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/group_norm.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/accumulate.h>
#include <torch/torch.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/group_norm_native.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward_native.h>
#include <ATen/ops/native_group_norm_native.h>
#endif

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
);

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
    // torch::Tensor& latent_offset,
    // torch::Tensor& patch_map,
    torch::Tensor& padding_idx);

void MockFuseGroupNormKernelImplInternal(
    const torch::Tensor& X,
    int N,
    int C,
    int H,
    int W,
    int group,
    torch::Tensor& Y,
    torch::Tensor& padding_idx);

torch::Tensor mock_groupnorm(const torch::Tensor& X, 
                int N, int C, int H, int W, 
                int group, 
                torch::Tensor& padding_idx) {
    const auto dtype = X.scalar_type();
    auto Y = torch::zeros({N, C, (H + 2), (W + 2)},  torch::TensorOptions().dtype(dtype).device((torch::kCUDA)));
    MockFuseGroupNormKernelImplInternal(X, N, C, H, W, group, Y, padding_idx);
    return Y;
}
torch::Tensor groupnorm(const torch::Tensor& X, 
                const c10::optional<torch::Tensor>& gamma_opt,
                const c10::optional<torch::Tensor>& beta_opt,
                int N, int C, int H, int W, 
                int group, double  eps, bool padding,
                torch::Tensor& latent_offset, 
                torch::Tensor& patch_map, 
                torch::Tensor& padding_idx) {
    const auto dtype = X.scalar_type();
    auto mean = torch::empty({N * C / group}, torch::TensorOptions().dtype(dtype).device((torch::kCUDA)));
    auto rstd = torch::empty({N * C / group},  torch::TensorOptions().dtype(dtype).device((torch::kCUDA)));
    int height = H;
    int width = W;
    if (padding) {
        height = H + 2;
        width = W + 2;
    }
    auto Y = torch::zeros({N, C, height, width},  torch::TensorOptions().dtype(dtype).device((torch::kCUDA)));
    c10::MaybeOwned<torch::Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
    const torch::Tensor& gamma = *gamma_maybe_owned;
    const torch::Tensor& beta = c10::value_or_else(beta_opt, [] { return torch::Tensor(); });
    GetMeanAndRstd(X, N, C, H, W, group, eps, mean, rstd, latent_offset, patch_map);
    
    FuseGroupNormKernelImplInternal(X, gamma, beta, N, C, H, W, group, padding, Y, mean, rstd, padding_idx);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("groupnorm", &groupnorm, "groupnorm");
    m.def("mock_groupnorm", &mock_groupnorm, "mock_groupnorm");
}