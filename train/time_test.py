"""Time full training step (forward + backward + muon)."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, VOCAB_SIZE,
    init_weights, forward_full, backward_full, param_size,
)
from train_big import encode, build_training_data, muon_step

corpus = build_training_data(target_size=200_000)
data = encode(corpus)
print(f"Data: {len(data)} tokens")

params = init_weights(seed=42)
print(f"Params: {param_size(params):,}")

m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}

TRAIN_CTX = 48
BATCH = 8
LR = 1e-4
max_start = len(data) - TRAIN_CTX - 1

# Warmup
for _ in range(3):
    starts = np.random.randint(0, max_start, size=BATCH)
    x = np.stack([data[s:s + TRAIN_CTX] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1:s + TRAIN_CTX + 1] for s in starts]).astype(np.int32)
    logits, cache = forward_full(params, x)
    grads = backward_full(params, logits, y, cache)
    muon_step(params, grads, m, v, 1, LR)

# Time
N = 30
start = time.time()
for step in range(N):
    starts = np.random.randint(0, max_start, size=BATCH)
    x = np.stack([data[s:s + TRAIN_CTX] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1:s + TRAIN_CTX + 1] for s in starts]).astype(np.int32)
    logits, cache = forward_full(params, x)
    grads = backward_full(params, logits, y, cache)
    muon_step(params, grads, m, v, step + 1, LR)
elapsed = time.time() - start
print(f"\n{N} steps @ ctx={TRAIN_CTX}, batch={BATCH}: {elapsed:.2f}s ({elapsed/N*1000:.0f}ms/step)")
