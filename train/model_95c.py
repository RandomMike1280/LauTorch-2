"""Pure NumPy Lau Transformer for training.

Architecture (must match the Lau inference very closely):
- vocab_size = 95 (all printable ASCII 32-126)
- d_model = 32
- n_layers = 2
- n_heads = 4 (head_dim = 8)
- d_ff = 128
- ctx_len = 64
- Tied input/output embeddings
- Pre-LayerNorm
- GELU activation (tanh approximation, fast)
- KV-cache (training does not use it; inference helpers do)
"""
import numpy as np

VOCAB_SIZE = 95
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'
D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 8
N_LAYERS = 2
D_FF = 128
CTX_LEN = 64
MAX_BATCH = 16


def gelu(x):
    # Tanh approximation of GELU, matches common transformer impl
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def gelu_grad(x):
    # Derivative of tanh approximation - needed for backward
    inner = 0.7978845608 * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    sech2 = 1.0 - t**2
    dinner = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner


def layernorm(x, gamma, beta, eps=1e-5):
    """x: (..., D). Returns (out, x_in, mean, var, x_norm)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = x_norm * gamma + beta
    return out, x, mean, var, x_norm


def layernorm_grad(dout, x_in, mean, var, x_norm, gamma, eps=1e-5):
    """Backward of layernorm. dout: gradient of L w.r.t. layernorm output.
    x_in: input to layernorm (pre-anything). x_norm: pre-gamma normalized.
    """
    N = x_in.shape[-1]
    std = np.sqrt(var + eps)
    dx_norm = dout * gamma  # (..., D)
    # dvar = sum_L (dout * gamma * (x - mean) * -0.5 * std^-3)
    dvar = np.sum(dx_norm * (x_in - mean) * (-0.5) * (var + eps)**(-1.5), axis=-1, keepdims=True)
    # dmean = sum_L dout * gamma * -1/std + dvar * sum(-2 * (x - mean)) / N
    dmean = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True) + \
            dvar * np.sum(-2.0 * (x_in - mean), axis=-1, keepdims=True) / N
    dx = dx_norm / std + dvar * 2.0 * (x_in - mean) / N + dmean / N
    dgamma = np.sum(dout * x_norm, axis=tuple(range(len(dout.shape) - 1)))
    dbeta = np.sum(dout, axis=tuple(range(len(dout.shape) - 1)))
    return dx, dgamma, dbeta


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets):
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
    return nll.mean(), log_probs, nll


def cross_entropy_grad(logits, targets):
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    probs[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
    probs /= (B * T)
    return probs


def init_weights(seed=0):
    rng = np.random.RandomState(seed)
    params = {}
    # Token embedding (tied with output)
    params['emb'] = rng.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.08
    # Per layer
    for i in range(N_LAYERS):
        params[f'l{i}.ln1_g'] = np.ones(D_MODEL, dtype=np.float32)
        params[f'l{i}.ln1_b'] = np.zeros(D_MODEL, dtype=np.float32)
        params[f'l{i}.ln2_g'] = np.ones(D_MODEL, dtype=np.float32)
        params[f'l{i}.ln2_b'] = np.zeros(D_MODEL, dtype=np.float32)
        params[f'l{i}.wq'] = (rng.randn(D_MODEL, D_MODEL) * 0.08).astype(np.float32)
        params[f'l{i}.wk'] = (rng.randn(D_MODEL, D_MODEL) * 0.08).astype(np.float32)
        params[f'l{i}.wv'] = (rng.randn(D_MODEL, D_MODEL) * 0.08).astype(np.float32)
        params[f'l{i}.wo'] = (rng.randn(D_MODEL, D_MODEL) * 0.08).astype(np.float32)
        params[f'l{i}.w1'] = (rng.randn(D_MODEL, D_FF) * 0.08).astype(np.float32)
        params[f'l{i}.b1'] = np.zeros(D_FF, dtype=np.float32)
        params[f'l{i}.w2'] = (rng.randn(D_FF, D_MODEL) * 0.08).astype(np.float32)
        params[f'l{i}.b2'] = np.zeros(D_MODEL, dtype=np.float32)
    # Final layer norm
    params['lnf_g'] = np.ones(D_MODEL, dtype=np.float32)
    params['lnf_b'] = np.zeros(D_MODEL, dtype=np.float32)
    return params


def forward(params, tokens, save_cache=True):
    """Forward pass. tokens: (B, T) of int ids. Returns logits (B, T, V)."""
    cache = {} if save_cache else None
    x = params['emb'][tokens]  # (B, T, D)
    if cache is not None:
        cache['emb_out'] = x
    for i in range(N_LAYERS):
        # LN1
        x_norm, x_raw, x_mean, x_var, x_post = layernorm(x, params[f'l{i}.ln1_g'], params[f'l{i}.ln1_b'])
        # Attention
        q = x_norm @ params[f'l{i}.wq']
        k = x_norm @ params[f'l{i}.wk']
        v = x_norm @ params[f'l{i}.wv']
        # Reshape to multi-head: (B, T, NH, HD) -> (B, NH, T, HD)
        B, T, D = q.shape
        q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        # Attention scores
        scale = 1.0 / np.sqrt(HEAD_DIM)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, NH, T, T)
        # Causal mask
        mask = np.tril(np.ones((T, T), dtype=np.float32))
        scores = scores * mask + (1.0 - mask) * (-1e9)
        attn = softmax(scores, axis=-1)
        attn_out = attn @ v  # (B, NH, T, HD)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_out = attn_out @ params[f'l{i}.wo']
        # Residual
        x = x + attn_out
        if cache is not None:
            cache[f'l{i}.x1'] = x
        # LN2
        x_norm2, x_raw2, x_mean2, x_var2, x_post2 = layernorm(x, params[f'l{i}.ln2_g'], params[f'l{i}.ln2_b'])
        # MLP
        h = x_norm2 @ params[f'l{i}.w1'] + params[f'l{i}.b1']
        h = gelu(h)
        h = h @ params[f'l{i}.w2'] + params[f'l{i}.b2']
        x = x + h
        if cache is not None:
            cache[f'l{i}.x2'] = x
    # Final LN
    x, x_raw, x_mean, x_var, x_post = layernorm(x, params['lnf_g'], params['lnf_b'])
    if cache is not None:
        cache['lnf_x'] = x
    # Tied embedding: logits = x @ emb.T
    logits = x @ params['emb'].T
    return logits, cache


def backward(params, tokens, logits, targets, cache):
    """Backward pass. Returns grads dict. cache has forward intermediates."""
    grads = {}
    B, T, V = logits.shape
    # Output gradient
    dlogits = cross_entropy_grad(logits, targets)
    # Tied embedding: x = cache['lnf_x'], logits = x @ emb.T
    x = cache['lnf_x']
    grads['emb'] = (dlogits.transpose(0, 2, 1) @ x).sum(axis=0)  # (V, D)
    dx = dlogits @ params['emb']  # (B, T, D)
    # Final LN
    dx, dgamma, dbeta = layernorm_grad(dx, cache['lnf_x'], cache['lnf_x'].mean(axis=-1, keepdims=True),
                                        cache['lnf_x'].var(axis=-1, keepdims=True),
                                        cache['lnf_x'], params['lnf_g'])
    grads['lnf_g'] = dgamma
    grads['lnf_b'] = dbeta
    # Back through layers
    for i in reversed(range(N_LAYERS)):
        # Residual + MLP
        x = cache[f'l{i}.x2']  # after MLP
        x_post_mlp = x  # save
        # dh: gradient through MLP
        # MLP: h = gelu(x_norm2 @ w1 + b1); mlp_out = h @ w2 + b2; x = x_residual + mlp_out
        # We need x_norm2 and h_relu from forward. We didn't cache them all. Re-derive.
        # For simplicity, re-run the forward of this layer (small model)
        x_res = cache[f'l{i}.x1']  # before MLP
        x_norm2 = (x_res - x_res.mean(axis=-1, keepdims=True)) / np.sqrt(x_res.var(axis=-1, keepdims=True) + 1e-5)
        x_norm2 = x_norm2 * params[f'l{i}.ln2_g'] + params[f'l{i}.ln2_b']
        h_pre = x_norm2 @ params[f'l{i}.w1'] + params[f'l{i}.b1']
        h_act = gelu(h_pre)
        # dx comes from two paths: residual and mlp
        # residual: contribution to x_res
        # mlp: contribution to x_residual + mlp_out
        # dx_residual = dx
        # h_grad = dx @ w2.T
        # d_w2 = h_act.T @ dx => for (B,T) flatten to (B*T, D_FF) @ (B*T, D)
        Bh, Th, Dh = dx.shape
        dh = dx @ params[f'l{i}.w2'].T  # (B, T, D_FF)
        dh_pre = dh * gelu_grad(h_pre)
        # dw1 = x_norm2.T @ dh_pre
        grads[f'l{i}.w2'] = (h_act.reshape(-1, D_FF).T @ dx.reshape(-1, D_MODEL))
        grads[f'l{i}.b2'] = dx.sum(axis=(0, 1))
        # dx_norm2 from MLP
        dx_norm2 = dh_pre @ params[f'l{i}.w1'].T
        grads[f'l{i}.w1'] = (x_norm2.reshape(-1, D_MODEL).T @ dh_pre.reshape(-1, D_FF))
        grads[f'l{i}.b1'] = dh_pre.sum(axis=(0, 1))
        # Back through LN2
        dx_res, dgamma2, dbeta2 = layernorm_grad(dx_norm2 + dx, x_res, x_res.mean(axis=-1, keepdims=True),
                                                 x_res.var(axis=-1, keepdims=True),
                                                 x_norm2 / 1.0, params[f'l{i}.ln2_g'])
        # Wait, layernorm_grad uses x_post which is the normalized output. Let's pass x_norm2 directly (post-gamma).
        # Actually we need to recompute properly. Let's just simplify and use the raw x_res.
        # Re-derive: x_norm2 = (x_res - mean) / std * gamma + beta
        # We want dgamma, dbeta wrt this. Use the layernorm_grad function with x_raw=x_res, post=x_norm2.
        # Hmm, layernorm_grad computes dgamma = sum(dout * x_norm). Let me just pass x_norm2.
        grads[f'l{i}.ln2_g'] = dgamma2
        grads[f'l{i}.ln2_b'] = dbeta2
        dx = dx_res
        # Now back through attention. Need to re-derive attn forward.
        x_norm1, x_mean1, x_var1 = x_res, x_res.mean(axis=-1, keepdims=True), x_res.var(axis=-1, keepdims=True)
        x_norm1 = (x_res - x_mean1) / np.sqrt(x_var1 + 1e-5)
        x_norm1 = x_norm1 * params[f'l{i}.ln1_g'] + params[f'l{i}.ln1_b']
        # Wait this is wrong. x_norm1 was the input to attention. cache[f'l{i}.x1'] is x after residual+attn.
        # Let me re-do this with a cleaner cache.
        pass  # Will be redone with cleaner cache
    return grads


# Cleaner full forward/backward with complete cache
def forward_full(params, tokens):
    """Forward with full cache for backward."""
    cache = {'tokens': tokens}
    x = params['emb'][tokens]
    cache['emb_out'] = x
    for i in range(N_LAYERS):
        # Save input to layer
        cache[f'l{i}.x_in'] = x
        # LN1
        x_mean = x.mean(axis=-1, keepdims=True)
        x_var = x.var(axis=-1, keepdims=True)
        x_norm = (x - x_mean) / np.sqrt(x_var + 1e-5)
        x_post = x_norm * params[f'l{i}.ln1_g'] + params[f'l{i}.ln1_b']
        cache[f'l{i}.ln1_mean'] = x_mean
        cache[f'l{i}.ln1_var'] = x_var
        cache[f'l{i}.ln1_xnorm'] = x_norm
        cache[f'l{i}.ln1_xpost'] = x_post
        # Attention
        q = x_post @ params[f'l{i}.wq']
        k = x_post @ params[f'l{i}.wk']
        v = x_post @ params[f'l{i}.wv']
        cache[f'l{i}.q'] = q
        cache[f'l{i}.k'] = k
        cache[f'l{i}.v'] = v
        B, T, D = q.shape
        q_h = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k_h = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v_h = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        scores = (q_h @ k_h.transpose(0, 1, 3, 2)) * scale
        mask = np.tril(np.ones((T, T), dtype=np.float32))
        scores_masked = scores * mask + (1.0 - mask) * (-1e9)
        attn = softmax(scores_masked, axis=-1)
        cache[f'l{i}.attn'] = attn
        cache[f'l{i}.scores_masked'] = scores_masked
        attn_out = attn @ v_h
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
        cache[f'l{i}.attn_concat'] = attn_out
        attn_proj = attn_out @ params[f'l{i}.wo']
        cache[f'l{i}.wo_out'] = attn_proj
        x = x + attn_proj
        cache[f'l{i}.x_res1'] = x
        # LN2
        x_mean2 = x.mean(axis=-1, keepdims=True)
        x_var2 = x.var(axis=-1, keepdims=True)
        x_norm2 = (x - x_mean2) / np.sqrt(x_var2 + 1e-5)
        x_post2 = x_norm2 * params[f'l{i}.ln2_g'] + params[f'l{i}.ln2_b']
        cache[f'l{i}.ln2_mean'] = x_mean2
        cache[f'l{i}.ln2_var'] = x_var2
        cache[f'l{i}.ln2_xnorm'] = x_norm2
        cache[f'l{i}.ln2_xpost'] = x_post2
        # MLP
        h_pre = x_post2 @ params[f'l{i}.w1'] + params[f'l{i}.b1']
        h_act = gelu(h_pre)
        cache[f'l{i}.h_pre'] = h_pre
        cache[f'l{i}.h_act'] = h_act
        h_out = h_act @ params[f'l{i}.w2'] + params[f'l{i}.b2']
        x = x + h_out
        cache[f'l{i}.x_out'] = x
    # Final LN
    x_mean_f = x.mean(axis=-1, keepdims=True)
    x_var_f = x.var(axis=-1, keepdims=True)
    x_norm_f = (x - x_mean_f) / np.sqrt(x_var_f + 1e-5)
    x_post_f = x_norm_f * params['lnf_g'] + params['lnf_b']
    cache['lnf_in'] = x
    cache['lnf_mean'] = x_mean_f
    cache['lnf_var'] = x_var_f
    cache['lnf_xnorm'] = x_norm_f
    cache['lnf_xpost'] = x_post_f
    # Tied embedding
    logits = x_post_f @ params['emb'].T
    return logits, cache


def backward_full(params, logits, targets, cache):
    grads = {}
    B, T, V = logits.shape
    # log-softmax grad
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    probs[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
    probs /= (B * T)
    dlogits = probs
    # Tied embedding backprop
    # logits = x_post_f @ emb.T, so d_emb = dlogits.T @ x_post_f, dx = dlogits @ emb
    grads['emb'] = (dlogits.transpose(0, 2, 1) @ cache['lnf_xpost']).sum(axis=0)
    dx = dlogits @ params['emb']
    # Final LN
    dxp, dgamma, dbeta = layernorm_grad(
        dx, cache['lnf_in'], cache['lnf_mean'], cache['lnf_var'],
        cache['lnf_xnorm'], params['lnf_g'])
    grads['lnf_g'] = dgamma
    grads['lnf_b'] = dbeta
    dx = dxp
    # Back through layers (reverse)
    for i in reversed(range(N_LAYERS)):
        # x_out = x_res1 + h_out, so:
        # dx_res1 = dx, dh_out = dx
        dx_res1 = dx
        dh_out = dx
        # Back through MLP h_out = h_act @ w2 + b2
        B, T, D_MODEL_ = dh_out.shape
        h_act_flat = cache[f'l{i}.h_act'].reshape(-1, D_FF)
        dh_out_flat = dh_out.reshape(-1, D_MODEL_)
        grads[f'l{i}.w2'] = h_act_flat.T @ dh_out_flat
        grads[f'l{i}.b2'] = dh_out.sum(axis=(0, 1))
        # dh_act = dh_out @ w2.T
        dh_act = (dh_out @ params[f'l{i}.w2'].T).reshape(B, T, D_FF)
        # dh_pre = dh_act * gelu_grad(h_pre)
        dh_pre = dh_act * gelu_grad(cache[f'l{i}.h_pre'])
        # h_pre = x_post2 @ w1 + b1
        x_post2_flat = cache[f'l{i}.ln2_xpost'].reshape(-1, D_MODEL_)
        dh_pre_flat = dh_pre.reshape(-1, D_FF)
        grads[f'l{i}.w1'] = x_post2_flat.T @ dh_pre_flat
        grads[f'l{i}.b1'] = dh_pre.sum(axis=(0, 1))
        # dx_post2 = dh_pre @ w1.T
        dx_post2 = (dh_pre @ params[f'l{i}.w1'].T)
        # dx_res1 += dx_post2 (LN2 input gradient)
        # Back through LN2
        dxp, dgamma2, dbeta2 = layernorm_grad(
            dx_post2 + dx_res1, cache[f'l{i}.x_res1'], cache[f'l{i}.ln2_mean'],
            cache[f'l{i}.ln2_var'], cache[f'l{i}.ln2_xnorm'], params[f'l{i}.ln2_g'])
        grads[f'l{i}.ln2_g'] = dgamma2
        grads[f'l{i}.ln2_b'] = dbeta2
        # dx = dxp (now this is the gradient w.r.t. layer input)
        dx = dxp
        # Back through attention
        # x_res1 = x_in + wo_out, so:
        dx_in = dx
        dwo_out = dx
        # wo_out = attn_concat @ wo
        attn_concat_flat = cache[f'l{i}.attn_concat'].reshape(-1, D_MODEL_)
        dwo_out_flat = dwo_out.reshape(-1, D_MODEL_)
        grads[f'l{i}.wo'] = attn_concat_flat.T @ dwo_out_flat
        # d_attn_concat = dwo_out @ wo.T
        d_attn_concat = (dwo_out @ params[f'l{i}.wo'].T)
        # attn_concat = attn @ v_h reshape-transpose
        # Back through reshape/transpose
        d_attn_concat_h = d_attn_concat.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        # attn_out = attn @ v_h
        attn = cache[f'l{i}.attn']
        d_attn = d_attn_concat_h @ cache[f'l{i}.v'].reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3).transpose(0, 1, 3, 2)
        dv_h = attn.transpose(0, 1, 3, 2) @ d_attn_concat_h
        # Back through softmax(scores_masked)
        # d_scores_masked = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
        d_scores_masked = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
        # Apply mask
        mask = np.tril(np.ones((T, T), dtype=np.float32))
        d_scores_masked = d_scores_masked * mask  # masked positions have 0 grad
        # scores = (q_h @ k_h.T) * scale
        scale = 1.0 / np.sqrt(HEAD_DIM)
        dq_h = d_scores_masked @ cache[f'l{i}.k'].reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3) * scale
        dk_h = d_scores_masked.transpose(0, 1, 3, 2) @ cache[f'l{i}.q'].reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3) * scale
        # Back through reshape
        dq = dq_h.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL_)
        dk = dk_h.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL_)
        dv = dv_h.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL_)
        # q = x_post @ wq, etc.
        x_post_flat = cache[f'l{i}.ln1_xpost'].reshape(-1, D_MODEL_)
        grads[f'l{i}.wq'] = x_post_flat.T @ dq.reshape(-1, D_MODEL_)
        grads[f'l{i}.wk'] = x_post_flat.T @ dk.reshape(-1, D_MODEL_)
        grads[f'l{i}.wv'] = x_post_flat.T @ dv.reshape(-1, D_MODEL_)
        # dx_post = dq @ wq.T + dk @ wk.T + dv @ wv.T
        dx_post = (dq @ params[f'l{i}.wq'].T) + (dk @ params[f'l{i}.wk'].T) + (dv @ params[f'l{i}.wv'].T)
        # Back through LN1
        dxp, dgamma1, dbeta1 = layernorm_grad(
            dx_post + dx_in, cache[f'l{i}.x_in'], cache[f'l{i}.ln1_mean'],
            cache[f'l{i}.ln1_var'], cache[f'l{i}.ln1_xnorm'], params[f'l{i}.ln1_g'])
        grads[f'l{i}.ln1_g'] = dgamma1
        grads[f'l{i}.ln1_b'] = dbeta1
        dx = dxp
        # Add to embedding grad for x_in (which is emb_out of prev layer or this token lookups)
        # We'll handle the embedding grad separately using dlogits path
    return grads


def param_size(params):
    total = sum(p.size for p in params.values())
    return total


def list_param_keys(params):
    return list(params.keys())
