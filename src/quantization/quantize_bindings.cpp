#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nf4_quant.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace py = pybind11;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

py::tuple quantize_nf4_py(py::array_t<float> input, int block_size = 64) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("input must be 1d array");
    }
    
    size_t n = buf.shape[0];
    if (n == 0) {
        throw std::runtime_error("input cannot be empty");
    }
    
    size_t output_bytes = (n + 1) / 2;
    size_t num_blocks = (n + block_size - 1) / block_size;
    
    float* d_input = nullptr;
    uint8_t* d_output = nullptr;
    float* d_scales = nullptr;
    
    try {
        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        CUDA_CHECK(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
        CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice));
        
        quantize_nf4_cuda(d_input, d_output, d_scales, n, block_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto quantized = py::array_t<uint8_t>(output_bytes);
        auto scales = py::array_t<float>(num_blocks);
        
        py::buffer_info quant_buf = quantized.request();
        py::buffer_info scale_buf = scales.request();
        
        CUDA_CHECK(cudaMemcpy(quant_buf.ptr, d_output, output_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scale_buf.ptr, d_scales, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_scales);
        
        return py::make_tuple(quantized, scales);
        
    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_scales) cudaFree(d_scales);
        throw;
    }
}

py::array_t<float> dequantize_nf4_py(py::array_t<uint8_t> input, py::array_t<float> scales, 
                                      size_t n, int block_size = 64) {
    py::buffer_info in_buf = input.request();
    py::buffer_info scale_buf = scales.request();
    
    if (in_buf.ndim != 1 || scale_buf.ndim != 1) {
        throw std::runtime_error("inputs must be 1d arrays");
    }
    
    if (n == 0) {
        throw std::runtime_error("n cannot be zero");
    }
    
    size_t num_blocks = scale_buf.shape[0];
    size_t expected_blocks = (n + block_size - 1) / block_size;
    
    if (num_blocks != expected_blocks) {
        throw std::runtime_error("scales array size mismatch");
    }
    
    uint8_t* d_input = nullptr;
    float* d_scales = nullptr;
    float* d_output = nullptr;
    
    try {
        CUDA_CHECK(cudaMalloc(&d_input, in_buf.size));
        CUDA_CHECK(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_input, in_buf.ptr, in_buf.size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales, scale_buf.ptr, num_blocks * sizeof(float), cudaMemcpyHostToDevice));
        
        dequantize_nf4_cuda(d_input, d_scales, d_output, n, block_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto output = py::array_t<float>(n);
        py::buffer_info out_buf = output.request();
        CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_input);
        cudaFree(d_scales);
        cudaFree(d_output);
        
        return output;
        
    } catch (...) {
        if (d_input) cudaFree(d_input);
        if (d_scales) cudaFree(d_scales);
        if (d_output) cudaFree(d_output);
        throw;
    }
}

PYBIND11_MODULE(campusgpt_quant, m) {
    m.doc() = "nf4 quantization kernels";
    
    m.def("quantize_nf4", &quantize_nf4_py, 
          "quantize tensor to nf4, returns (quantized_data, scales)",
          py::arg("input"), py::arg("block_size") = 64);
    
    m.def("dequantize_nf4", &dequantize_nf4_py, 
          "dequantize nf4 tensor back to float",
          py::arg("input"), py::arg("scales"), py::arg("n"), py::arg("block_size") = 64);
}
