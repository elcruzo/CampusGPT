# CampusGPT

fine-tuned llama 2 on howard-specific campus Q&A using qlora. handles questions about registration, dining, housing, IT, etc.

includes custom C/CUDA components for performance (optional).

## setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your HF token
```

### optional: build native components

faster tokenizer (C) and quantization kernels (CUDA):

```bash
# tokenizer only (no gpu needed)
python setup.py build_ext --inplace

# quantization kernels (requires CUDA)
mkdir build && cd build
cmake .. && make
```

benchmarks (tested on M1):
- tokenizer: 5.3x faster than transformers (0.007s vs 0.036s on 1k sentences)
- quantization: 1.6x faster than bitsandbytes (requires cuda)

see `src/tokenizer/README.md` and `src/quantization/README.md`.

## training

```bash
# quick test with small model (cpu)
python train_simple.py

# llama 2 with qlora (needs gpu or colab)
python train_llama_qlora.py

# or use notebooks/train_llama_colab.ipynb on colab
```

## serving

```bash
python -m src.api.main
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what are the library hours?"}'
```

## data

20 seed examples in `data/raw/campus_qa.jsonl`. format:

```json
{"instruction": "question", "input": "optional context", "output": "answer", "category": "academic"}
```

expand with your own campus data for better results.

## notes

- uses peft/lora (r=16, alpha=32) to keep training memory under 8gb
- fastapi server with basic caching
- eval metrics in src/evaluation
