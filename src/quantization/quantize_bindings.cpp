#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nf4_quant.h"
#include <cuda_runtime.h>

namespace py = pybind11;

py::array_t<uint8_t> quantize_nf4_py(py::array_t<float> input, int block_size = 64) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("input must be 1d array");
    }
    
    size_t n = buf.shape[0];
    size_t output_bytes = (n + 1) / 2;
    size_t num_blocks = (n + block_size - 1) / block_size;
    
    // allocate device memory
    float* d_input;
    uint8_t* d_output;
    float* d_scales;
    
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_scales, num_blocks * sizeof(float));
    
    // copy input to device
    cudaMemcpy(d_input, buf.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // quantize
    float* h_scales = new float[num_blocks];
    quantize_nf4_cuda(d_input, d_output, d_scales, n, block_size);
    
    // copy results back
    auto output = py::array_t<uint8_t>(output_bytes);
    py::buffer_info out_buf = output.request();
    cudaMemcpy(out_buf.ptr, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scales, d_scales, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
    delete[] h_scales;
    
    return output;
}

py::array_t<float> dequantize_nf4_py(py::array_t<uint8_t> input, py::array_t<float> scales, 
                                      size_t n, int block_size = 64) {
    py::buffer_info in_buf = input.request();
    py::buffer_info scale_buf = scales.request();
    
    size_t num_blocks = scale_buf.shape[0];
    
    // allocate device memory
    uint8_t* d_input;
    float* d_scales;
    float* d_output;
    
    cudaMalloc(&d_input, in_buf.size);
    cudaMalloc(&d_scales, num_blocks * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    // copy to device
    cudaMemcpy(d_input, in_buf.ptr, in_buf.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scale_buf.ptr, num_blocks * sizeof(float), cudaMemcpyHostToDevice);
    
    // dequantize
    dequantize_nf4_cuda(d_input, d_scales, d_output, n, block_size);
    
    // copy results back
    auto output = py::array_t<float>(n);
    py::buffer_info out_buf = output.request();
    cudaMemcpy(out_buf.ptr, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(d_input);
    cudaFree(d_scales);
    cudaFree(d_output);
    
    return output;
}

PYBIND11_MODULE(campusgpt_quant, m) {
    m.doc() = "nf4 quantization kernels";
    
    m.def("quantize_nf4", &quantize_nf4_py, "quantize tensor to nf4",
          py::arg("input"), py::arg("block_size") = 64);
    
    m.def("dequantize_nf4", &dequantize_nf4_py, "dequantize nf4 tensor",
          py::arg("input"), py::arg("scales"), py::arg("n"), py::arg("block_size") = 64);
}
