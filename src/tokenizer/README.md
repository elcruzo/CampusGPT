# fast tokenizer

BPE tokenizer implemented in C. faster than transformers tokenizer on CPU.

## build

```bash
python setup.py build_ext --inplace
```

## usage

```python
from campusgpt_tokenizer import FastTokenizer

tok = FastTokenizer("vocab.txt")
tokens = tok.encode("hello world")
text = tok.decode(tokens)
```

## implementation

- mmap for fast vocab loading
- binary search for token lookup
- greedy longest-match encoding
- no malloc in hot path (pre-allocated buffers)
- zero-copy where possible

## benchmark

on test dataset (1000 sentences):
- fast_tokenizer: ~0.1s
- transformers: ~0.3s
- speedup: ~3x

actual speedup depends on vocab size and text length. bigger wins on longer texts.

## limitations

- simplified BPE (no byte-level encoding yet)
- vocab must be line-delimited text file
- no special token handling
- basic unicode support

good enough for prototyping. would need more work for production.
