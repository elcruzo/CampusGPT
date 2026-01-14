import numpy as np
import sys

try:
    import campusgpt_quant
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False
    print("quantization kernels not built (need CUDA)")
    sys.exit(0)

def test_roundtrip():
    data = np.random.randn(1024).astype(np.float32)
    
    quantized, scales = campusgpt_quant.quantize_nf4(data)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data))
    
    mse = np.mean((data - recovered) ** 2)
    max_err = np.max(np.abs(data - recovered))
    
    print(f"roundtrip mse: {mse:.6f}")
    print(f"roundtrip max error: {max_err:.6f}")
    
    assert mse < 0.1, f"mse too high: {mse}"
    print("✓ roundtrip test passed")

def test_compression():
    data = np.random.randn(1024).astype(np.float32)
    
    quantized, scales = campusgpt_quant.quantize_nf4(data)
    
    original_bytes = data.nbytes
    compressed_bytes = quantized.nbytes + scales.nbytes
    ratio = original_bytes / compressed_bytes
    
    print(f"original: {original_bytes} bytes")
    print(f"compressed: {compressed_bytes} bytes")
    print(f"ratio: {ratio:.2f}x")
    
    assert ratio > 1.5, f"compression ratio too low: {ratio}"
    print("✓ compression test passed")

def test_accuracy():
    data = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]).astype(np.float32)
    
    quantized, scales = campusgpt_quant.quantize_nf4(data)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data))
    
    for i, (orig, rec) in enumerate(zip(data, recovered)):
        error = abs(orig - rec)
        print(f"  val[{i}]: orig={orig:.3f} rec={rec:.3f} err={error:.3f}")
    
    mse = np.mean((data - recovered) ** 2)
    assert mse < 0.05, f"accuracy mse too high: {mse}"
    print("✓ accuracy test passed")

def test_different_sizes():
    sizes = [64, 256, 1024, 4096, 16384]
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        quantized, scales = campusgpt_quant.quantize_nf4(data)
        recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data))
        
        mse = np.mean((data - recovered) ** 2)
        assert mse < 0.1, f"mse too high for size {size}: {mse}"
        
        expected_bytes = (size + 1) // 2
        assert quantized.nbytes == expected_bytes, f"wrong size for {size}"
    
    print("✓ different sizes test passed")

def test_block_sizes():
    data = np.random.randn(1024).astype(np.float32)
    block_sizes = [32, 64, 128]
    
    for bs in block_sizes:
        quantized, scales = campusgpt_quant.quantize_nf4(data, block_size=bs)
        recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data), block_size=bs)
        
        mse = np.mean((data - recovered) ** 2)
        expected_num_blocks = (len(data) + bs - 1) // bs
        
        assert len(scales) == expected_num_blocks, f"wrong scales count for bs={bs}"
        assert mse < 0.15, f"mse too high for block_size={bs}: {mse}"
    
    print("✓ block sizes test passed")

def test_edge_cases():
    zeros = np.zeros(128).astype(np.float32)
    quantized, scales = campusgpt_quant.quantize_nf4(zeros)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(zeros))
    assert np.allclose(recovered, zeros, atol=0.01), "zeros failed"
    
    ones = np.ones(128).astype(np.float32)
    quantized, scales = campusgpt_quant.quantize_nf4(ones)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(ones))
    mse = np.mean((ones - recovered) ** 2)
    assert mse < 0.05, f"ones mse too high: {mse}"
    
    single = np.array([0.5]).astype(np.float32)
    quantized, scales = campusgpt_quant.quantize_nf4(single)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, 1)
    assert abs(single[0] - recovered[0]) < 0.2, "single value failed"
    
    odd = np.random.randn(127).astype(np.float32)
    quantized, scales = campusgpt_quant.quantize_nf4(odd)
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(odd))
    mse = np.mean((odd - recovered) ** 2)
    assert mse < 0.1, f"odd size mse too high: {mse}"
    
    print("✓ edge cases test passed")

def test_determinism():
    data = np.random.randn(512).astype(np.float32)
    
    q1, s1 = campusgpt_quant.quantize_nf4(data)
    q2, s2 = campusgpt_quant.quantize_nf4(data)
    
    assert np.array_equal(q1, q2), "quantization not deterministic"
    assert np.array_equal(s1, s2), "scales not deterministic"
    
    print("✓ determinism test passed")

if __name__ == "__main__":
    test_roundtrip()
    test_compression()
    test_accuracy()
    test_different_sizes()
    test_block_sizes()
    test_edge_cases()
    test_determinism()
    print("\nall quantization tests passed")
