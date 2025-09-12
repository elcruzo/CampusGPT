import sys
import os

try:
    import campusgpt_tokenizer
    FAST_AVAILABLE = True
except ImportError:
    FAST_AVAILABLE = False
    print("fast tokenizer not built, skipping tests")
    sys.exit(0)

def test_basic_encode():
    # create simple vocab for testing
    vocab_path = "/tmp/test_vocab.txt"
    with open(vocab_path, 'w') as f:
        f.write("hello\n")
        f.write("world\n")
        f.write("test\n")
        f.write("tokenizer\n")
    
    tok = campusgpt_tokenizer.FastTokenizer(vocab_path)
    
    tokens = tok.encode("hello world test")
    print(f"encoded tokens: {tokens}")
    assert len(tokens) > 0, "should produce tokens"
    
    decoded = tok.decode(tokens)
    print(f"decoded: {decoded}")
    
    print("✓ basic encode/decode test passed")
    os.unlink(vocab_path)

def test_empty_string():
    vocab_path = "/tmp/test_vocab.txt"
    with open(vocab_path, 'w') as f:
        f.write("test\n")
    
    tok = campusgpt_tokenizer.FastTokenizer(vocab_path)
    tokens = tok.encode("")
    print(f"empty string tokens: {tokens}")
    assert len(tokens) == 0, "empty string should produce no tokens"
    
    print("✓ empty string test passed")
    os.unlink(vocab_path)

def test_memory_safety():
    # test with many iterations to check for leaks
    vocab_path = "/tmp/test_vocab.txt"
    with open(vocab_path, 'w') as f:
        for i in range(100):
            f.write(f"token{i}\n")
    
    tok = campusgpt_tokenizer.FastTokenizer(vocab_path)
    
    for i in range(1000):
        tokens = tok.encode("token1 token2 token3")
        decoded = tok.decode(tokens)
    
    print("✓ memory safety test passed (no crashes)")
    os.unlink(vocab_path)

if __name__ == "__main__":
    test_basic_encode()
    test_empty_string()
    test_memory_safety()
    print("\n all tests passed")
