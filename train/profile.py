"""Profile forward+backward to find the bottleneck."""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, VOCAB_SIZE,
    init_weights, forward_full, backward_full, param_size,
)

params = init_weights(seed=42)
print(f"Params: {param_size(params):,}")

TRAIN_CTX = 48
BATCH = 8

# Fake data
np.random.seed(0)
x = np.random.randint(0, VOCAB_SIZE, size=(BATCH, TRAIN_CTX)).astype(np.int32)
y = np.random.randint(0, VOCAB_SIZE, size=(BATCH, TRAIN_CTX)).astype(np.int32)

# Warmup
for _ in range(3):
    logits, cache = forward_full(params, x)
    grads = backward_full(params, logits, y, cache)

N = 20
# Time forward
start = time.time()
for _ in range(N):
    logits, cache = forward_full(params, x)
elapsed = time.time() - start
print(f"Forward: {elapsed/N*1000:.1f}ms/step")

# Time backward
logits, cache = forward_full(params, x)
start = time.time()
for _ in range(N):
    grads = backward_full(params, logits, y, cache)
elapsed = time.time() - start
print(f"Backward: {elapsed/N*1000:.1f}ms/step")

# Time forward+backward
start = time.time()
for _ in range(N):
    logits, cache = forward_full(params, x)
    grads = backward_full(params, logits, y, cache)
elapsed = time.time() - start
print(f"Forward+Backward: {elapsed/N*1000:.1f}ms/step")
