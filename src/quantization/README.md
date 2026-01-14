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

weights = np.random.randn(4096).astype(np.float32)

# quantize - returns (quantized_data, scales)
quantized, scales = campusgpt_quant.quantize_nf4(weights, block_size=64)

# dequantize back to float
recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(weights), block_size=64)

# check error
mse = np.mean((weights - recovered) ** 2)
print(f"mse: {mse:.6f}")
print(f"compression: {weights.nbytes / (quantized.nbytes + scales.nbytes):.1f}x")
```

## implementation

- block-wise quantization (default block_size=64)
- absmax scaling per block
- NF4 lookup table for quantization
- race-free CUDA kernels (one thread per output byte)
- 2 values packed per byte
- proper error handling with CUDA_CHECK macros

## algorithm

NF4 maps floats to 4-bit using a non-linear lookup table optimized for normal distributions:

1. compute absmax per block
2. normalize values by absmax
3. find closest entry in NF4 table
4. pack 2 4-bit values per byte

## tests

```bash
python test_quantization.py
```

tests roundtrip accuracy, compression ratio, different sizes/block sizes, edge cases, and determinism.

## limitations

- requires CUDA gpu
- no double quantization
- fixed block size per call

for production use cases, consider bitsandbytes which has more optimizations.
