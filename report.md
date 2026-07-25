# LauTorch Inference — Final Report

## Overview
A Lau-inference implementation of a tiny GPT-2 transformer language model trained in Python. The model is exported to Lau as a 2-layer, 16-dim, 4-head, char-level transformer with KV-cache and greedy decoding.

## File Layout (4 files, all under 5000 bytes)
- `chat.lau` (4642 bytes) — inference engine
- `weights_emb.laum` (3285 bytes) — embedding matrix (96 × 16)
- `weights_l0.laum` (3652 bytes) — layer 0 weights (12 tensors)
- `weights_l1.laum` (3716 bytes) — layer 1 weights + final LayerNorm (14 tensors)

## Model Architecture
- Vocabulary: 96 (printable ASCII chars, char-level tokenizer)
- d_model: 16
- Heads: 4 (head dim 4)
- Layers: 2
- d_ff: 16 (feed-forward)
- Context length: 16
- Tied input/output embeddings
- Pre-LayerNorm, GELU activation, softmax attention
- INT8 symmetric per-tensor quantization
- KV-cache with sliding window for autoregressive generation

## Math Customizations (Lau has no built-ins)
- `sq(x)` — math.sqrt via 8-iteration Newton-Raphson
- `th(x)` — math.tanh via rational approximation (Pade-like)
- `ex(x)` — math.exp via Taylor series + reciprocation for negative x
- `d(s, p)` — base-95 to int8 decoder (reads 2 chars, maps to -127..127)

## Run
```
lau chat.lau
```

## Known Limitations
- Model is intentionally tiny (4832 parameters) for speed and to fit embeds
- Output is somewhat garbled — the model has learned a Caesar-shift-like pattern
- Per-token statement count is high (~100K) due to repeated string decoding
- Suggested: run with `--statement-cost 0` for fast interactive demo
