#include "nf4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__constant__ float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

__device__ inline uint8_t quantize_value(float val, float scale) {
    float normalized = (scale > 1e-8f) ? val / scale : 0.0f;
    normalized = fmaxf(-1.0f, fminf(1.0f, normalized));
    
    uint8_t best_idx = 0;
    float best_diff = fabsf(normalized - NF4_TABLE[0]);
    
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(normalized - NF4_TABLE[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    
    return best_idx;
}

__device__ inline float dequantize_value(uint8_t idx, float scale) {
    return NF4_TABLE[idx & 0xF] * scale;
}

__global__ void compute_scales_kernel(const float* input, float* scales, 
                                       size_t n, int block_size) {
    int block_idx = blockIdx.x;
    int start = block_idx * block_size;
    int end = min(start + block_size, (int)n);
    
    if (start >= n) return;
    
    float local_max = 0.0f;
    for (int i = start; i < end; i++) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    scales[block_idx] = local_max;
}

__global__ void quantize_nf4_kernel(const float* input, uint8_t* output, 
                                     const float* scales, size_t n, int block_size) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_bytes = (n + 1) / 2;
    
    if (byte_idx >= output_bytes) return;
    
    int idx0 = byte_idx * 2;
    int idx1 = byte_idx * 2 + 1;
    
    int block_idx0 = idx0 / block_size;
    float scale0 = scales[block_idx0];
    uint8_t q0 = (idx0 < n) ? quantize_value(input[idx0], scale0) : 0;
    
    uint8_t q1 = 0;
    if (idx1 < n) {
        int block_idx1 = idx1 / block_size;
        float scale1 = scales[block_idx1];
        q1 = quantize_value(input[idx1], scale1);
    }
    
    output[byte_idx] = q0 | (q1 << 4);
}

__global__ void dequantize_nf4_kernel(const uint8_t* input, const float* scales,
                                       float* output, size_t n, int block_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= n) return;
    
    int block_idx = global_idx / block_size;
    float scale = scales[block_idx];
    
    int byte_idx = global_idx / 2;
    int nibble = global_idx % 2;
    
    uint8_t packed = input[byte_idx];
    uint8_t q = (nibble == 0) ? (packed & 0xF) : (packed >> 4);
    
    output[global_idx] = dequantize_value(q, scale);
}

void quantize_nf4_cuda(const float* input, uint8_t* output, float* scales,
                       size_t n, int block_size) {
    if (n == 0) return;
    
    int num_quant_blocks = (n + block_size - 1) / block_size;
    compute_scales_kernel<<<num_quant_blocks, 1>>>(input, scales, n, block_size);
    cudaDeviceSynchronize();
    
    size_t output_bytes = (n + 1) / 2;
    int threads = 256;
    int grid = (output_bytes + threads - 1) / threads;
    
    quantize_nf4_kernel<<<grid, threads>>>(input, output, scales, n, block_size);
    cudaDeviceSynchronize();
}

void dequantize_nf4_cuda(const uint8_t* input, const float* scales, float* output,
                         size_t n, int block_size) {
    if (n == 0) return;
    
    int threads = 256;
    int grid = (n + threads - 1) / threads;
    
    dequantize_nf4_kernel<<<grid, threads>>>(input, scales, output, n, block_size);
    cudaDeviceSynchronize();
}
