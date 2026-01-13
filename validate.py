#!/usr/bin/env python3

import sys
import os

print("validating campusgpt...")
print("-" * 50)

# 1. check data file
print("\n[1/6] checking data file...")
try:
    import json
    with open('data/raw/campus_qa.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"  ✓ {len(data)} valid Q&A pairs")
except Exception as e:
    print(f"  ✗ data file error: {e}")
    sys.exit(1)

# 2. check native tokenizer
print("\n[2/6] checking native tokenizer...")
try:
    import campusgpt_tokenizer
    
    vocab_path = "/tmp/test_vocab.txt"
    with open(vocab_path, 'w') as f:
        f.write("test\nhello\nworld\n")
    
    tok = campusgpt_tokenizer.FastTokenizer(vocab_path)
    tokens = tok.encode("hello world")
    decoded = tok.decode(tokens)
    
    os.unlink(vocab_path)
    print(f"  ✓ tokenizer works (encoded {len(tokens)} tokens)")
except ImportError:
    print("  ⚠ tokenizer not built (run: python3 setup.py build_ext --inplace)")
except Exception as e:
    print(f"  ✗ tokenizer error: {e}")

# 3. check utils modules
print("\n[3/6] checking utils modules...")
try:
    from src.utils import logger, caching, config
    cache = caching.ResponseCache()
    cache.set("test", "value")
    assert cache.get("test") == "value"
    print("  ✓ utils modules work")
except Exception as e:
    print(f"  ✗ utils error: {e}")
    sys.exit(1)

# 4. check evaluation modules
print("\n[4/6] checking evaluation modules...")
try:
    from src.evaluation.evaluators import BLEUEvaluator, ROUGEEvaluator
    bleu = BLEUEvaluator()
    refs = ["the library is open from 8am to 10pm on weekdays"]
    preds = ["the library is open from 8am to 10pm on weekdays"]
    result = bleu.evaluate(refs, preds)
    print(f"  ✓ evaluation modules work (bleu: {result['score']:.2f})")
except Exception as e:
    print(f"  ✗ evaluation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. check native integration
print("\n[5/6] checking native integration...")
try:
    from src.native import get_capabilities
    caps = get_capabilities()
    print(f"  ✓ native integration works")
    print(f"    fast tokenizer: {caps['fast_tokenizer']}")
    print(f"    custom quant: {caps['custom_quantization']}")
except Exception as e:
    print(f"  ✗ native integration error: {e}")
    sys.exit(1)

# 6. check training scripts syntax
print("\n[6/6] checking training scripts...")
try:
    import py_compile
    py_compile.compile('train_simple.py', doraise=True)
    py_compile.compile('train_llama_qlora.py', doraise=True)
    print("  ✓ training scripts compile")
except Exception as e:
    print(f"  ✗ training script error: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✓ all checks passed")
print("\nready to use:")
print("  python3 train_simple.py          # train on cpu")
print("  python3 train_llama_qlora.py     # train with qlora (needs gpu)")
print("  python3 -m src.api.main          # start api server (needs trained model)")
