#include "nf4_quant.h"
#include <cuda_runtime.h>
#include <stdexcept>

extern "C" {

void* cuda_malloc(size_t bytes) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    return ptr;
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed");
    }
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H failed");
    }
}

}  // extern "C"
