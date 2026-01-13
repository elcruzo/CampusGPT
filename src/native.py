# optional native components with fallback

import sys
import os

# ensure project root is in path (where .so files are)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import campusgpt_tokenizer
    FAST_TOKENIZER_AVAILABLE = True
except ImportError:
    FAST_TOKENIZER_AVAILABLE = False
    campusgpt_tokenizer = None

try:
    import campusgpt_quant
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    campusgpt_quant = None

def get_tokenizer(vocab_file=None):
    """get tokenizer, preferring fast C version if available"""
    if FAST_TOKENIZER_AVAILABLE and vocab_file:
        print("using fast C tokenizer")
        return campusgpt_tokenizer.FastTokenizer(vocab_file)
    else:
        print("using transformers tokenizer")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("gpt2")  # fallback

def quantize_tensor(tensor, block_size=64):
    """quantize tensor, using custom kernels if available"""
    if QUANTIZATION_AVAILABLE:
        import numpy as np
        if hasattr(tensor, 'cpu'):  # torch tensor
            data = tensor.cpu().numpy().astype(np.float32)
        else:
            data = tensor
        return campusgpt_quant.quantize_nf4(data, block_size)
    else:
        # fallback to bitsandbytes or return original
        try:
            import bitsandbytes as bnb
            return bnb.functional.quantize_nf4(tensor)
        except ImportError:
            return tensor  # no quantization available

def get_capabilities():
    """return dict of available native components"""
    return {
        'fast_tokenizer': FAST_TOKENIZER_AVAILABLE,
        'custom_quantization': QUANTIZATION_AVAILABLE,
    }

if __name__ == "__main__":
    caps = get_capabilities()
    print("native components:")
    print(f"  fast tokenizer: {caps['fast_tokenizer']}")
    print(f"  custom quantization: {caps['custom_quantization']}")
