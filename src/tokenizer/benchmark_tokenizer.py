import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import campusgpt_tokenizer
    FAST_AVAILABLE = True
except ImportError as e:
    FAST_AVAILABLE = False
    print(f"fast tokenizer not available: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("transformers not available for comparison")

def generate_test_sentences(n=1000):
    sentences = []
    words = ["hello", "world", "test", "tokenizer", "fast", "python", "code", "campus", "gpt"]
    for i in range(n):
        sentence = " ".join(words[j % len(words)] for j in range(i % 20 + 5))
        sentences.append(sentence)
    return sentences

def benchmark_fast_tokenizer(sentences):
    vocab_path = "/tmp/bench_vocab.txt"
    with open(vocab_path, 'w') as f:
        for word in ["hello", "world", "test", "tokenizer", "fast", "python", "code", "campus", "gpt"]:
            f.write(f"{word}\n")
    
    tok = campusgpt_tokenizer.FastTokenizer(vocab_path)
    
    start = time.time()
    for sent in sentences:
        tokens = tok.encode(sent)
    elapsed = time.time() - start
    
    import os
    os.unlink(vocab_path)
    
    return elapsed

def benchmark_hf_tokenizer(sentences):
    if not HF_AVAILABLE:
        return None
    
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        
        start = time.time()
        for sent in sentences:
            tokens = tok.encode(sent)
        elapsed = time.time() - start
        
        return elapsed
    except:
        return None

if __name__ == "__main__":
    print("generating test data...")
    sentences = generate_test_sentences(1000)
    
    print(f"benchmarking on {len(sentences)} sentences...")
    
    fast_time = benchmark_fast_tokenizer(sentences)
    print(f"fast tokenizer: {fast_time:.3f}s")
    
    if HF_AVAILABLE:
        hf_time = benchmark_hf_tokenizer(sentences)
        if hf_time:
            print(f"huggingface:    {hf_time:.3f}s")
            speedup = hf_time / fast_time
            print(f"speedup:        {speedup:.2f}x")
        else:
            print("huggingface benchmark failed")
    else:
        print("huggingface not available for comparison")
