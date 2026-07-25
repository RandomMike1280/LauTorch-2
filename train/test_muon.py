"""Measure how slow Muon is vs pure AdamW."""
import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import init_weights, param_size

params = init_weights(seed=42)
print(f"Params: {param_size(params):,}")

grads = {k: np.random.randn(*p.shape).astype(np.float32) * 0.01 for k, p in params.items()}
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}


def _zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(np.float32)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (np.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def muon_step_test():
    for k in params:
        g = grads[k]
        if g.ndim >= 2:
            m[k] = 0.95 * m[k] + g
            update = _zeropower_via_newtonschulz5(m[k])

# Warm up
muon_step_test()

N = 50
start = time.time()
for _ in range(N):
    muon_step_test()
elapsed = time.time() - start
print(f"50 muon steps: {elapsed:.3f}s ({elapsed/N*1000:.1f}ms/step)")

# But w1 is 64x256, which is the biggest. Let's see per-tensor.
for name, p in params.items():
    if p.ndim >= 2:
        g = grads[name]
        start = time.time()
        for _ in range(20):
            update = _zeropower_via_newtonschulz5(g)
        elapsed = time.time() - start
        print(f"  {name:20s} {str(p.shape):20s} NS5: {elapsed/20*1000:.2f}ms")
