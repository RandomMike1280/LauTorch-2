"""Sanity training loop: confirm that loss decreases monotonically after the LN fix.

Runs 50 SGD steps on a fixed batch and prints loss per step.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    D_MODEL, D_FF, N_HEADS, HEAD_DIM, N_LAYERS, init_weights,
    forward_full, backward_full,
)


def main():
    np.random.seed(0)
    params = init_weights()
    B, T = 2, 8
    tokens = np.random.randint(0, 95, size=(B, T)).astype(np.int64)
    targets = np.random.randint(0, 95, size=(B, T)).astype(np.int64)

    def loss_fn(p):
        l, _ = forward_full(p, tokens)
        log_probs = l - l.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
        return -log_probs[np.arange(B)[:, None], np.arange(T)[None, :], targets].mean()

    L0 = loss_fn(params)
    print(f"Initial loss: {L0:.4f}")

    lr = 1e-2
    prev = L0
    losses = [L0]
    for step in range(1, 51):
        logits, cache = forward_full(params, tokens)
        grads = backward_full(params, logits, targets, cache)
        for k in grads:
            params[k] -= lr * grads[k]
        L = loss_fn(params)
        losses.append(L)
        if L > prev + 1e-9:
            print(f"  step {step:2d} loss={L:.4f}  *** INCREASE from {prev:.4f}")
        prev = L
        if step % 5 == 0:
            print(f"  step {step:2d} loss={L:.4f}")

    final = losses[-1]
    print(f"Final loss: {final:.4f}")
    print(f"Total reduction: {L0 - final:.4f} ({100 * (L0 - final) / L0:.1f}%)")

    # Monotonicity check (allow occasional non-monotonic small bump)
    monotonic = all(losses[i] >= losses[i+1] - 1e-3 for i in range(len(losses) - 1))
    if monotonic:
        print("Loss is monotonically decreasing (within 1e-3 noise).")
    else:
        print("WARNING: Loss has noisy increases.")


if __name__ == "__main__":
    main()
