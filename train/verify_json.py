"""Verify a trained Lau Transformer loads from weights.json and answers the 3 test prompts.

This script:
1. Loads weights.json
2. Reconstructs the int8 weights with their scales
3. Runs forward passes on the 3 test prompts
4. Reports test accuracy
"""
import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, VOCAB_SIZE,
    forward_full, init_weights,
)

# 64-char vocab matching train_big.py
_PUNCT = ".,!?;:\"'()-+=/*<>@#$%^&_~`"
VOCAB_CHARS = ['\n', ' '] + [str(i) for i in range(10)] + [chr(ord('a') + i) for i in range(26)] + list(_PUNCT)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB_CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}


def encode(text):
    out = []
    for ch in text:
        if ch == '\r':
            ch = '\n'
        if ch in CHAR_TO_ID:
            out.append(CHAR_TO_ID[ch])
        else:
            out.append(2)
    return np.array(out, dtype=np.int64)


def decode(ids):
    return ''.join(ID_TO_CHAR[int(i)] for i in ids)


def load_weights_json(path):
    """Load weights.json and reconstruct the params dict (dequantized to float32)."""
    with open(path, 'r') as f:
        data = json.load(f)
    cfg = data['config']
    assert cfg['d_model'] == D_MODEL, f"d_model mismatch: {cfg['d_model']} vs {D_MODEL}"
    assert cfg['n_layers'] == N_LAYERS, f"n_layers mismatch: {cfg['n_layers']} vs {N_LAYERS}"
    assert cfg['n_heads'] == N_HEADS, f"n_heads mismatch: {cfg['n_heads']} vs {N_HEADS}"
    assert cfg['vocab_size'] == VOCAB_SIZE, f"vocab_size mismatch: {cfg['vocab_size']} vs {VOCAB_SIZE}"

    names = data['names']
    shapes = data['shapes']
    scales = data['scales']
    flat = np.array(data['data'], dtype=np.int8)

    # Reconstruct each tensor
    params = {}
    offset = 0
    for name, shape, scale in zip(names, shapes, scales):
        n = 1
        for d in shape:
            n *= d
        chunk = flat[offset:offset + n]
        offset += n
        w = chunk.astype(np.float32) * scale
        params[name] = w.reshape(shape)
    assert offset == len(flat), f"data length mismatch: {offset} vs {len(flat)}"
    return params, cfg


def generate(params, prompt, max_new=50, temperature=0.0, ctx_len=CTX_LEN):
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        ctx = tokens[-ctx_len:]
        x = np.array([ctx], dtype=np.int32)
        logits, _ = forward_full(params, x)
        last_logits = logits[0, -1, :]
        if temperature <= 0:
            next_id = int(last_logits.argmax())
        else:
            scaled = last_logits / temperature
            scaled = scaled - scaled.max()
            probs = np.exp(scaled)
            probs = probs / probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        tokens.append(next_id)
        if next_id == 0:
            break
        if len(tokens) > 400:
            break
    return decode(tokens)


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'weights.json')
    print(f"Loading weights from {weights_path}...")
    params, cfg = load_weights_json(weights_path)
    print(f"Loaded {len(params)} tensors, d_model={cfg['d_model']}, n_layers={cfg['n_layers']}")

    test_prompts = [
        ("human: hello\nbot:", "hello"),
        ("human: what's 1+1?\nbot:", "2"),
        ("human: what's the capital of france?\nbot:", "paris"),
    ]

    print("\n=== Test prompts (greedy decode) ===")
    passed = 0
    for prompt, must_contain in test_prompts:
        out = generate(params, prompt, max_new=30, temperature=0.0)
        ok = must_contain in out.lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {prompt!r}")
        print(f"         -> {out!r} (must contain {must_contain!r})")
    print(f"\nScore: {passed}/{len(test_prompts)}")


if __name__ == "__main__":
    main()
