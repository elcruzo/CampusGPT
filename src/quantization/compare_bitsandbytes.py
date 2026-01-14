import numpy as np
import time
import sys

try:
    import campusgpt_quant
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False
    print("custom quantization not available")

try:
    import bitsandbytes as bnb
    import torch
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("bitsandbytes not available")

if not CUSTOM_AVAILABLE and not BNB_AVAILABLE:
    print("no quantization libraries available")
    sys.exit(1)

def benchmark_custom(data, iterations=100):
    if not CUSTOM_AVAILABLE:
        return None, None
    
    start = time.time()
    for _ in range(iterations):
        quantized, scales = campusgpt_quant.quantize_nf4(data)
    elapsed = time.time() - start
    
    recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data))
    mse = np.mean((data - recovered) ** 2)
    
    return elapsed / iterations, mse

def benchmark_bnb(data, iterations=100):
    if not BNB_AVAILABLE:
        return None, None
    
    tensor = torch.from_numpy(data).cuda()
    
    start = time.time()
    for _ in range(iterations):
        quantized, state = bnb.functional.quantize_nf4(tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    recovered = bnb.functional.dequantize_nf4(quantized, state)
    mse = torch.mean((tensor - recovered) ** 2).item()
    
    return elapsed / iterations, mse

def compare_accuracy(data):
    print("\naccuracy comparison:")
    print(f"{'method':<15} {'mse':<15} {'max_err':<15}")
    print("-" * 45)
    
    if CUSTOM_AVAILABLE:
        quantized, scales = campusgpt_quant.quantize_nf4(data)
        recovered = campusgpt_quant.dequantize_nf4(quantized, scales, len(data))
        mse = np.mean((data - recovered) ** 2)
        max_err = np.max(np.abs(data - recovered))
        print(f"{'custom':<15} {mse:<15.6f} {max_err:<15.6f}")
    
    if BNB_AVAILABLE:
        tensor = torch.from_numpy(data).cuda()
        quantized, state = bnb.functional.quantize_nf4(tensor)
        recovered = bnb.functional.dequantize_nf4(quantized, state)
        mse = torch.mean((tensor - recovered) ** 2).item()
        max_err = torch.max(torch.abs(tensor - recovered)).item()
        print(f"{'bitsandbytes':<15} {mse:<15.6f} {max_err:<15.6f}")

if __name__ == "__main__":
    sizes = [1024, 4096, 16384, 65536]
    
    print("benchmarking quantization...")
    print(f"{'size':<10} {'custom (ms)':<15} {'bnb (ms)':<15} {'speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        
        custom_time, custom_mse = benchmark_custom(data)
        bnb_time, bnb_mse = benchmark_bnb(data)
        
        if custom_time and bnb_time:
            speedup = bnb_time / custom_time
            print(f"{size:<10} {custom_time*1000:<15.3f} {bnb_time*1000:<15.3f} {speedup:<10.2f}x")
        elif custom_time:
            print(f"{size:<10} {custom_time*1000:<15.3f} {'N/A':<15} {'N/A':<10}")
        elif bnb_time:
            print(f"{size:<10} {'N/A':<15} {bnb_time*1000:<15.3f} {'N/A':<10}")
    
    data = np.random.randn(4096).astype(np.float32)
    compare_accuracy(data)
