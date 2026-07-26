"""Numerical gradient check for forward_full + backward_full.

Verifies that gradients computed by backward_full match finite-difference
numerical gradients for every parameter in the model, including:
- token embedding (tied)
- LN1 and LN2 scale (gamma) and bias (beta) for both layers
- attention weights wq, wk, wv, wo for both layers
- MLP weights w1, b1, w2, b2 for both layers
- final LN gamma and beta
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    D_MODEL, D_FF, N_HEADS, HEAD_DIM, N_LAYERS, init_weights,
    forward_full, backward_full,
)


def loss_fn(p, tokens, targets):
    """Mean cross-entropy loss."""
    logits, _ = forward_full(p, tokens)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    B, T = logits.shape[:2]
    return -log_probs[np.arange(B)[:, None], np.arange(T)[None, :], targets].mean()


def numerical_grad(p, key, indices, loss, eps, ndim):
    """Compute gradient of `loss` w.r.t. `p[key]` at the given indices.

    `indices` is a list of (i, j) tuples. For 1D params, only `i` is used.
    """
    out = np.zeros(len(indices))
    for k, (i, j) in enumerate(indices):
        p_plus = {kk: v.copy() for kk, v in p.items()}
        p_minus = {kk: v.copy() for kk, v in p.items()}
        if ndim == 1:
            p_plus[key][i] += eps
            p_minus[key][i] -= eps
        else:
            p_plus[key][i, j] += eps
            p_minus[key][i, j] -= eps
        out[k] = (loss(p_plus) - loss(p_minus)) / (2 * eps)
    return out


def analytic_grad_at(grad, indices, ndim):
    """Extract analytic gradient values at the given indices."""
    if ndim == 1:
        return np.array([grad[i] for (i, _) in indices])
    return np.array([grad[i, j] for (i, j) in indices])


def check_parameter(p, key, analytic_grad, tokens, targets, eps, picks):
    """Compare analytic vs numerical gradients for a single parameter."""
    loss = lambda pp: loss_fn(pp, tokens, targets)
    ndim = p[key].ndim
    num = numerical_grad(p, key, picks, loss, eps, ndim)
    ana = analytic_grad_at(analytic_grad, picks, ndim)
    rel = np.abs(num - ana) / np.maximum(np.maximum(np.abs(num), np.abs(ana)), 1e-8)
    rel_max = rel.max()
    tag = "OK" if rel_max < 1e-2 else "FAIL"
    print(f"  {key} (eps={eps}): max_rel = {rel_max:.4e}  {tag}")
    for k, (i, j) in enumerate(picks):
        print(f"    [{i},{j}]  num={num[k]:+.6e}  ana={ana[k]:+.6e}  rel={rel[k]:.4e}")
    return rel_max


def main():
    np.random.seed(0)
    params = init_weights()
    B, T = 1, 4
    tokens = np.random.randint(0, 95, size=(B, T)).astype(np.int64)
    targets = np.random.randint(0, 95, size=(B, T)).astype(np.int64)

    # Analytic backward
    logits, cache = forward_full(params, tokens)
    grads = backward_full(params, logits, targets, cache)

    # Pick a few entries per parameter
    keys_to_check = []
    for i in range(N_LAYERS):
        keys_to_check.extend([
            f'l{i}.ln1_g', f'l{i}.ln1_b',
            f'l{i}.ln2_g', f'l{i}.ln2_b',
            f'l{i}.wq', f'l{i}.wk', f'l{i}.wv', f'l{i}.wo',
            f'l{i}.w1', f'l{i}.b1', f'l{i}.w2', f'l{i}.b2',
        ])
    keys_to_check.extend(['emb', 'lnf_g', 'lnf_b'])

    # Sample picks per parameter
    def picks_for(key):
        p = params[key]
        if p.ndim == 1:
            idx = sorted(set([0, p.shape[0] // 2, p.shape[0] - 1]))
            return [(i, 0) for i in idx]
        # 2D
        a, b = p.shape
        idx_a = sorted(set([0, a // 4, a // 2, 3 * a // 4, a - 1]))
        idx_b = sorted(set([0, b // 4, b // 2, 3 * b // 4, b - 1]))
        return [(i, j) for i in idx_a for j in idx_b]

    # Run with two epsilons
    for eps in [1e-3, 1e-4]:
        print(f"\n========== eps = {eps} ==========")
        all_ok = True
        for key in keys_to_check:
            picks = picks_for(key)
            rel_max = check_parameter(params, key, grads[key], tokens, targets, eps, picks)
            if rel_max >= 1e-2:
                all_ok = False
        if all_ok:
            print(f"\nAll gradients match (eps={eps}).")
        else:
            print(f"\nSome gradients failed at eps={eps}.")


if __name__ == "__main__":
    main()
