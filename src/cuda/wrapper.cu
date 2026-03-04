#define ENABLE_FP32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// Define global device properties as expected by llmc kernels
// Define global device properties as expected by llmc kernels
// deviceProp is declared extern in cuda_common.h
cudaDeviceProp deviceProp;

// cublaslt_handle is defined in cublas_common.h, so we don't need to define it.
// cublas_handle is NOT defined in cublas_common.h, so we MUST define it here.
cublasHandle_t cublas_handle;

#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "encoder.cuh"
#include "matmul.cuh" 
// matmul.cuh includes gelu.cuh, so we don't need to include it again
#include "layernorm.cuh"
#include "attention.cuh"
#include "fused_classifier.cuh"
#include "adamw.cuh"
#include "global_norm.cuh"

extern "C" {
    void init_cuda_c() {
        int deviceIdx = 0;
        cudaCheck(cudaSetDevice(deviceIdx));
        cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
        
        cublasCheck(cublasCreate(&cublas_handle));
        cublasCheck(cublasLtCreate(&cublaslt_handle));
        
        // Allocate workspace for cublasLt
        cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
    }

    void cleanup_cuda_c() {
        if (cublaslt_workspace != NULL) {
            cudaCheck(cudaFree(cublaslt_workspace));
        }
        cublasCheck(cublasDestroy(cublas_handle));
        cublasCheck(cublasLtDestroy(cublaslt_handle));
    }

    // Memory management wrappers
    void malloc_c(void** ptr, size_t size) {
        cudaCheck(cudaMalloc(ptr, size));
    }

    void free_c(void* ptr) {
        cudaCheck(cudaFree(ptr));
    }

    void memcpy_htod_c(void* dest, const void* src, size_t size) {
        cudaCheck(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
    }

    void memcpy_dtoh_c(void* dest, const void* src, size_t size) {
        cudaCheck(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
    }

    void memset_c(void* dest, int value, size_t size) {
        cudaCheck(cudaMemset(dest, value, size));
    }
    
    void device_synchronize_c() {
        cudaCheck(cudaDeviceSynchronize());
    }

    // Kernel wrappers
    void encoder_forward_c(float* out, const int* inp, float* wte, float* wpe, int B, int T, int C) {
        encoder_forward((floatX*)out, inp, (floatX*)wte, (floatX*)wpe, B, T, C, 0);
    }

    void encoder_backward_c(float* dwte, float* dwpe, float* dout, const int* inp, const int* inputs_cpu, 
                           float* scratch, int* workload_indices, int* bucket_info, 
                           int B, int T, int C, unsigned int seed) {
        encoder_backward((floatX*)dwte, (floatX*)dwpe, (floatX*)scratch, 
                        workload_indices, (int4*)bucket_info, (floatX*)dout, inp, inputs_cpu, B, T, C, seed, 0);
    }

    void layernorm_forward_c(float* out, float* mean, float* rstd,
                            float* inp, float* weight, float* bias,
                            int B, int T, int C) {
        layernorm_forward((floatX*)out, mean, rstd, (floatX*)inp, (floatX*)weight, (floatX*)bias, B, T, C, 0);
    }

    void layernorm_backward_c(float* dinp, float* dweight, float* dbias, float* scratch,
                             float* dout, float* inp, float* weight, float* mean, float* rstd,
                             int B, int T, int C) {
        layernorm_backward((floatX*)dinp, (floatX*)dweight, (floatX*)dbias, (floatX*)scratch, 
                           (floatX*)dout, (floatX*)inp, (floatX*)weight, mean, rstd, B, T, C, 0);
    }

    void matmul_forward_c(float* out, float* inp, float* weight, float* bias, 
                         int B, int T, int C, int OC) {
        matmul_forward_cublaslt((floatX*)out, (floatX*)inp, (floatX*)weight, (floatX*)bias, B, T, C, OC, 0);
    }

    void matmul_backward_c(float* dinp, float* dweight, float* dbias,
                          float* dout, float* inp, float* weight,
                          float* dbias_buffer,
                          int B, int T, int C, int OC) {
        matmul_backward((floatX*)dinp, (floatX*)dweight, (floatX*)dbias, 
                        (floatX*)dout, (floatX*)inp, (floatX*)weight, 
                        (floatX*)dbias_buffer, B, T, C, OC, 0);
    }

    void attention_forward_c(float* out, float* qkvr, float* att, float* scratch,
                            int B, int T, int C, int NH) {
        // Wrapper for standard attention (not CUDNN for now to simplify)
        // attention_forward signature: (out, qkvr, att, inp, B, T, C, NH, stream)
        // We use 'scratch' as 'inp' because attention_forward reuses it as scratch.
        attention_forward((floatX*)out, (floatX*)qkvr, (floatX*)att, (floatX*)scratch, B, T, C, NH, 0);
    }

    void attention_backward_c(float* dinp, float* dqkvr, float* datt, float* scratch,
                             float* dout, float* qkvr, float* att,
                             int B, int T, int C, int NH) {
        attention_backward((floatX*)dinp, (floatX*)dqkvr, (floatX*)datt, (floatX*)scratch,
                           (floatX*)dout, (floatX*)qkvr, (floatX*)att,
                           B, T, C, NH, 0);
    }

    void fused_classifier_c(float* dlogits, float* losses, float* logits,
                           const int* targets, int B, int T, int V, int Vp) {
        // dloss is 1.0/(B*T) for mean loss
        float dloss = 1.0f / (B * T);
        fused_classifier((floatX*)logits, losses, dloss, targets, B, T, V, Vp, std::false_type{}, 0);
    }

    void fused_classifier_backward_c(float* dlogits, float* losses, 
                                    const int* targets, int B, int T, int V, int Vp) {
        float dloss = 1.0f / (B * T);
        // fused_classifier backward updates logits in-place with gradients
        // here dlogits points to the same memory as logits (acts.output)
        fused_classifier((floatX*)dlogits, losses, dloss, targets, B, T, V, Vp, std::true_type{}, 0);
    }
    
    // Fused residual forward for blocks
    void fused_residual_forward5_c(float* residual_out, float* ln_out, float* mean, float* rstd,
                                  float* residual_in, float* ln_in, 
                                  float* weight, float* bias,
                                  int N, int C) {
        fused_residual_forward5((floatX*)residual_out, (floatX*)ln_out, mean, rstd,
                                (floatX*)residual_in, (floatX*)ln_in,
                                (floatX*)weight, (floatX*)bias,
                                N, C, 0);
    }

    void residual_forward_c(float* out, const float* inp1, const float* inp2, int N) {
        // Simple element-wise addition to avoid dealing with CUDA kernels, or we could launch a custom kernel
        // Actually best to use llmc's residual_forward
        // Since we didn't find a direct residual kernel launcher in layernorm.cuh that strictly matches residual_forward, let's write a simple kernel
        // wait, we can just use residual_forward defined in train_gpt2.c for now? No, must be GPU.
        // Actually, train_gpt2.cu has `residual_forward_kernel` inside `layernorm.cuh`
        // Wait, layernorm.cuh contains `residual_forward_kernel`, let's just use cublas / custom
        cudaCheck(cudaMemcpy(out, inp1, N * sizeof(float), cudaMemcpyDeviceToDevice));
        float alpha = 1.0f;
        cublasCheck(cublasSaxpy(cublas_handle, N, &alpha, inp2, 1, out, 1));
    }

    void residual_backward_c(float* dinp1, float* dinp2, const float* dout, int N) {
        // dinp1 += dout, dinp2 += dout
        float alpha = 1.0f;
        cublasCheck(cublasSaxpy(cublas_handle, N, &alpha, dout, 1, dinp1, 1));
        cublasCheck(cublasSaxpy(cublas_handle, N, &alpha, dout, 1, dinp2, 1));
    }

    void accumulate_c(float* a, const float* b, size_t n) {
        float alpha = 1.0f;
        cublasCheck(cublasSaxpy(cublas_handle, (int)n, &alpha, b, 1, a, 1));
    }

    void gelu_forward_c(float* out, const float* inp, size_t N) {
        gelu_forward((floatX*)out, (floatX*)inp, N, 0);
    }

    void gelu_backward_c(float* dinp, const float* inp, const float* dout, size_t N) {
        if (dinp != dout) {
            cudaCheck(cudaMemcpy(dinp, dout, N * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        gelu_backward_inplace((floatX*)dinp, (floatX*)inp, N, 0);
    }

    // AdamW step
    void adamw_step_c(float* params, float* grads, float* m, float* v, 
                     float learning_rate, float beta1, float beta2, float eps, float weight_decay,
                     int step, size_t num_parameters) {
        // Assuming params and grads are float (FP32) since ENABLE_FP32 is defined
        // 1.0f is grad_scale
        adamw_update<float, float>(params, NULL, grads, m, v, num_parameters, 
                                   1, 1, 1, 1, // strides (assumed contiguous for now)
                                   learning_rate, beta1, beta2, step, eps, weight_decay,
                                   1.0f, step, 0);
    }

    void global_norm_c(float* out, const float* values, size_t count) {
        // We use a single slice (num_slices = 1) and assume count is total params
        // max_num_block_sums is calculated in a similar way to get_max_num_block_sums
        const int block_size = 512;
        const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
        int max_num_block_sums = grid_size; // since num_slices = 1
        
        global_norm_squared<float>(out, values, count, 0, 1, max_num_block_sums, true, 0);
        global_norm_aggregate_kernel<<<1, 1024, 0, 0>>>(out, max_num_block_sums);
    }

    void scale_grads_c(float* grads, float scale, size_t count) {
        cublasCheck(cublasSscal(cublas_handle, (int)count, &scale, grads, 1));
    }
}
