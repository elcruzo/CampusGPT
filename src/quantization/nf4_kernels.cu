#include "nf4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// NF4 lookup table (normalized float 4-bit)
__constant__ float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

__device__ inline uint8_t quantize_value(float val, float scale) {
    float normalized = val / (scale + 1e-8f);
    normalized = fmaxf(-1.0f, fminf(1.0f, normalized));
    
    // find closest value in table
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

__global__ void quantize_nf4_kernel(const float* input, uint8_t* output, 
                                     float* scales, size_t n, int block_size) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    int global_idx = block_idx * block_size + tid;
    
    if (global_idx >= n) return;
    
    // compute block scale (absmax)
    __shared__ float block_max;
    
    if (tid == 0) {
        float local_max = 0.0f;
        for (int i = 0; i < block_size && (block_idx * block_size + i) < n; i++) {
            local_max = fmaxf(local_max, fabsf(input[block_idx * block_size + i]));
        }
        block_max = local_max;
        scales[block_idx] = local_max;
    }
    __syncthreads();
    
    // quantize values
    if (global_idx < n) {
        uint8_t q = quantize_value(input[global_idx], block_max);
        
        // pack 2 values per byte
        int byte_idx = global_idx / 2;
        int nibble = global_idx % 2;
        
        if (nibble == 0) {
            atomicAnd(&output[byte_idx], 0xF0);  // clear lower nibble
            atomicOr(&output[byte_idx], q);       // set lower nibble
        } else {
            atomicAnd(&output[byte_idx], 0x0F);  // clear upper nibble
            atomicOr(&output[byte_idx], q << 4); // set upper nibble
        }
    }
}

__global__ void dequantize_nf4_kernel(const uint8_t* input, const float* scales,
                                       float* output, size_t n, int block_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= n) return;
    
    int block_idx = global_idx / block_size;
    float scale = scales[block_idx];
    
    // unpack 2 values per byte
    int byte_idx = global_idx / 2;
    int nibble = global_idx % 2;
    
    uint8_t packed = input[byte_idx];
    uint8_t q = (nibble == 0) ? (packed & 0xF) : (packed >> 4);
    
    output[global_idx] = dequantize_value(q, scale);
}

void quantize_nf4_cuda(const float* input, uint8_t* output, float* scales,
                       size_t n, int block_size) {
    int num_blocks = (n + block_size - 1) / block_size;
    int threads = min(block_size, 256);
    
    // zero output
    size_t output_bytes = (n + 1) / 2;
    cudaMemset(output, 0, output_bytes);
    
    quantize_nf4_kernel<<<num_blocks, threads>>>(input, output, scales, n, block_size);
    cudaDeviceSynchronize();
}

void dequantize_nf4_cuda(const uint8_t* input, const float* scales, float* output,
                         size_t n, int block_size) {
    int num_blocks = (n + 255) / 256;
    int threads = 256;
    
    dequantize_nf4_kernel<<<num_blocks, threads>>>(input, scales, output, n, block_size);
    cudaDeviceSynchronize();
}
