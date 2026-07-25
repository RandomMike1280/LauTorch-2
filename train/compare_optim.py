"""Quick comparison: AdamW vs Muon at lr=1e-4, 200 steps."""
import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, VOCAB_SIZE,
    init_weights, forward_full, backward_full, param_size,
)
from train_big import encode, build_training_data

corpus = build_training_data(target_size=200_000)
data = encode(corpus)

TRAIN_CTX = 48
BATCH = 8
max_start = len(data) - TRAIN_CTX - 1

def run_test(optimizer_name, lr, n_steps=200):
    np.random.seed(42)
    params = init_weights(seed=42)
    m = {k: np.zeros_like(p) for k, p in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        starts = np.random.randint(0, max_start, size=BATCH)
        x = np.stack([data[s:s + TRAIN_CTX] for s in starts]).astype(np.int32)
        y = np.stack([data[s + 1:s + TRAIN_CTX + 1] for s in starts]).astype(np.int32)
        logits, cache = forward_full(params, x)
        log_probs = logits - logits.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
        nll = -log_probs[np.arange(BATCH)[:, None], np.arange(TRAIN_CTX)[None, :], y]
        loss = nll.mean()
        losses.append(float(loss))
        grads = backward_full(params, logits, y, cache)
        if optimizer_name == 'adam':
            for k in params:
                g = grads[k]
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.95 * v[k] + 0.05 * (g * g)
                params[k] -= lr * (m[k] / (np.sqrt(v[k]) + 1e-8))
        elif optimizer_name == 'muon':
            for k in params:
                g = grads[k]
                if g.ndim >= 2:
                    m[k] = 0.95 * m[k] + g
                    X = m[k].astype(np.float32)
                    if X.shape[0] > X.shape[1]:
                        X = X.T
                    X = X / (np.linalg.norm(X) + 1e-7)
                    a, b, c = (3.4445, -4.7750, 2.0315)
                    for _ in range(5):
                        A = X @ X.T
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X
                    if m[k].shape[0] > m[k].shape[1]:
                        X = X.T
                    scale = 0.2 * math.sqrt(max(m[k].shape[0], m[k].shape[1]))
                    params[k] -= lr * scale * X
                else:
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.95 * v[k] + 0.05 * (g * g)
                    params[k] -= lr * (m[k] / (np.sqrt(v[k]) + 1e-8))
    elapsed = time.time() - t0
    print(f"{optimizer_name} lr={lr}: {n_steps} steps in {elapsed:.1f}s ({elapsed/n_steps*1000:.0f}ms/step)")
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    return losses

run_test('adam', 1e-4, 1000)
run_test('adam', 1e-3, 1000)
run_test('muon', 1e-4, 1000)
run_test('muon', 1e-3, 1000)
run_test('muon', 1e-2, 1000)
