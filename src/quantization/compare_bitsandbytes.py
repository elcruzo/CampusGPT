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
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("bitsandbytes not available")

if not CUSTOM_AVAILABLE and not BNB_AVAILABLE:
    print("no quantization libraries available")
    sys.exit(1)

def benchmark_custom(data, iterations=100):
    if not CUSTOM_AVAILABLE:
        return None
    
    start = time.time()
    for _ in range(iterations):
        quantized = campusgpt_quant.quantize_nf4(data)
    elapsed = time.time() - start
    
    return elapsed / iterations

def benchmark_bnb(data, iterations=100):
    if not BNB_AVAILABLE:
        return None
    
    import torch
    tensor = torch.from_numpy(data).cuda()
    
    start = time.time()
    for _ in range(iterations):
        quantized = bnb.functional.quantize_nf4(tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations

if __name__ == "__main__":
    sizes = [1024, 4096, 16384, 65536]
    
    print("benchmarking quantization...")
    print(f"{'size':<10} {'custom (ms)':<15} {'bnb (ms)':<15} {'speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        
        custom_time = benchmark_custom(data)
        bnb_time = benchmark_bnb(data)
        
        if custom_time and bnb_time:
            speedup = bnb_time / custom_time
            print(f"{size:<10} {custom_time*1000:<15.3f} {bnb_time*1000:<15.3f} {speedup:<10.2f}x")
        elif custom_time:
            print(f"{size:<10} {custom_time*1000:<15.3f} {'N/A':<15} {'N/A':<10}")
        elif bnb_time:
            print(f"{size:<10} {'N/A':<15} {bnb_time*1000:<15.3f} {'N/A':<10}")
