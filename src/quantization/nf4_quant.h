#ifndef NF4_QUANT_H
#define NF4_QUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NF4 quantization: map fp16 to 4-bit
void quantize_nf4_cuda(const float* input, uint8_t* output, float* scales, 
                       size_t n, int block_size);

// NF4 dequantization: map 4-bit back to fp16
void dequantize_nf4_cuda(const uint8_t* input, const float* scales, float* output,
                         size_t n, int block_size);

#ifdef __cplusplus
}
#endif

#endif
