# nf4 quantization

CUDA kernels for NF4 (Normal Float 4-bit) quantization. compresses model weights from 32-bit to 4-bit.

## build

requires CUDA toolkit:

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make
```

## usage

```python
import campusgpt_quant
import numpy as np

# quantize weights
weights = np.random.randn(4096).astype(np.float32)
quantized = campusgpt_quant.quantize_nf4(weights, block_size=64)

# 8x compression (32-bit to 4-bit)
print(f"compression: {weights.nbytes / quantized.nbytes:.1f}x")
```

## implementation

- block-wise quantization (default block_size=64)
- absmax scaling per block
- NF4 lookup table for quantization
- optimized CUDA kernels with coalesced memory access
- 2 values packed per byte

## algorithm

NF4 maps floats to 4-bit using a non-linear lookup table optimized for normal distributions:

1. compute absmax per block
2. normalize values by absmax
3. find closest entry in NF4 table
4. pack 2 4-bit values per byte

## benchmark

on T4 GPU, quantizing 65k floats:
- custom kernels: ~0.05ms
- bitsandbytes: ~0.08ms
- speedup: ~1.6x

actual performance depends on tensor size and block size. bigger blocks = faster but less accurate.

## limitations

- requires CUDA gpu
- simplified NF4 (basic implementation)
- no double quantization yet
- fixed block size

good for understanding quantization internals. production code should use bitsandbytes.
