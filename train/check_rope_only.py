"""Verify RoPE only: bypass the broken LN backward and test pure RoPE gradient flow."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    D_MODEL, D_FF, N_HEADS, HEAD_DIM, N_LAYERS, init_weights,
    forward_full, backward_full, compute_rope_freqs,
    apply_rope_heads, rope_backward_heads,
)


def test_rope_forward_backward_consistency():
    """Test 1: For a random RoPE rotation, <forward(x), dy> should equal <x, backward(dy)>."""
    np.random.seed(0)
    T = 6
    cos, sin = compute_rope_freqs(T, HEAD_DIM)
    x = np.random.randn(N_HEADS, T, HEAD_DIM).astype(np.float32)
    dy = np.random.randn(N_HEADS, T, HEAD_DIM).astype(np.float32)
    y = apply_rope_heads(x[None], cos, sin)[0]
    dx = rope_backward_heads(dy[None], cos, sin)[0]
    inner_product_y_dy = (y * dy).sum()
    inner_product_x_dx = (x * dx).sum()
    err = abs(inner_product_y_dy - inner_product_x_dx)
    print(f"Test 1: RoPE orthogonality. <y, dy> = {inner_product_y_dy:.6f}, <x, dx> = {inner_product_x_dx:.6f}, err = {err:.6e}")
    assert err < 1e-3, "RoPE backward is inconsistent with forward."


def test_rope_in_attention():
    """Test 2: Compare attention scores with and without RoPE.

    With RoPE, attn[b,h,t1,t2] = q_h[b,h,t1] @ k_h[b,h,t2], where q_h, k_h are RoPE-rotated.
    This should equal (rot(q)[t1]) @ (rot(k)[t2]), where rot is the position-dependent rotation.
    """
    np.random.seed(1)
    T = 4
    H = N_HEADS
    D = HEAD_DIM
    cos, sin = compute_rope_freqs(T, D)
    q = np.random.randn(H, T, D).astype(np.float32)
    k = np.random.randn(H, T, D).astype(np.float32)
    q_h = apply_rope_heads(q[None], cos, sin)[0]
    k_h = apply_rope_heads(k[None], cos, sin)[0]

    # Manually compute rotational inner product between positions p and q
    # <R(p q), R(q k)> = <q, k> (rotation preserves dot product)
    inner_at_dist = np.zeros((H, T, T))
    for h in range(H):
        for p in range(T):
            for q_pos in range(T):
                inner_at_dist[h, p, q_pos] = (q_h[h, p] * k_h[h, q_pos]).sum()

    # Check: inner product should equal q @ k when p == q_pos (same position -> same rotation)
    for h in range(H):
        for p in range(T):
            qk = (q[h, p] * k[h, p]).sum()
            err = abs(inner_at_dist[h, p, p] - qk)
            assert err < 1e-4, f"RoPE at same position should preserve dot product: got {err}"
    print("Test 2: RoPE preserves dot product at the same position. OK")


def test_rope_position_dependence():
    """Test 3: Verify that RoPE encodes relative position via the rotation."""
    np.random.seed(2)
    T = 5
    H = 1
    D = HEAD_DIM
    cos, sin = compute_rope_freqs(T, D)

    # Pick a single q and k vector and broadcast to all positions
    q = np.random.randn(H, 1, D).astype(np.float32)
    k = np.random.randn(H, 1, D).astype(np.float32)
    q = np.broadcast_to(q, (H, T, D)).copy()
    k = np.broadcast_to(k, (H, T, D)).copy()

    # Compute attention scores for all positions of q
    q_h = apply_rope_heads(q, cos, sin)  # (H, T, D)
    k_h = apply_rope_heads(k, cos, sin)

    # Score(s, t) = q at position s dot k at position t
    scores = np.zeros((T, T))
    for s in range(T):
        for t in range(T):
            scores[s, t] = (q_h[0, s] * k_h[0, t]).sum()

    # Verify that the relative-position term is correct:
    # score(s, t) = q^T R^T(s) R(t) k = q^T R(t-s) k
    # We can verify: R^T(s) R(t) = R(t - s)
    # So scores[s, t] = q^T R(t) k  (when q is at position 0)
    # But q is at position 0 in our setup since we only have 1 anchor.
    # Actually q_h[0, s] = R(s) q. And k_h[0, t] = R(t) k.
    # So scores[s, t] = q^T R(s)^T R(t) k = q^T R(t-s) k

    # Reference: compute R(t-s) q, then dot with k
    from model import compute_rope_freqs as crf
    cos2, sin2 = crf(T, D)
    # For each (s, t), apply rotation by (t-s) to q
    delta = np.zeros((T, T, D))
    for s in range(T):
        for t in range(T):
            # R(t-s) k
            d = t - s
            # Recompute cos/sin at position d
            c_d, s_d = crf(1, D)
            # Use rope for one position
            # Actually let me use the helper on a single vector
            pass
    # Simpler: just check one diagonal entry
    print("Test 3: Same-dot-product check at varying distances OK")
    print(f"  scores[0,0] = {scores[0,0]:.4f}, scores[1,1] = {scores[1,1]:.4f}, scores[2,2] = {scores[2,2]:.4f}")
    print(f"  scores[0,3] = {scores[0,3]:.4f}, scores[1,4] = {scores[1,4]:.4f}")
    # relative-position property: scores[s, t] depends only on (t-s)
    for delta in range(-2, 3):
        # Find all (s, t) with t-s = delta
        pairs = [(s, s + delta) for s in range(T) if 0 <= s + delta < T]
        if len(pairs) > 1:
            vals = [scores[s, t] for s, t in pairs]
            err = max(abs(v - vals[0]) for v in vals)
            print(f"  delta={delta}: vals={vals}, err={err:.4e}")
            assert err < 1e-4, f"RoPE should give same score for same delta"


if __name__ == "__main__":
    test_rope_forward_backward_consistency()
    test_rope_in_attention()
    test_rope_position_dependence()
    print("\nAll RoPE tests passed.")
