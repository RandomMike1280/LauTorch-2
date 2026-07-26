"""Train medium model with CORRECT 95-char tokenization and export to weights.json."""
import os, sys, time, json, math
import numpy as np
sys.path.insert(0, 'train')
from model import (VOCAB_SIZE, VOCAB, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM,
                   D_FF, CTX_LEN, init_weights, forward_full, backward_full, param_size)
from train import encode, get_batch, get_lr

# ---------------- Muon optimizer (2D matrices) + AdamW (1D params) ----------------
# Mirrors train/train_big.py implementation, ported here so we don't pull in its
# data builder. Newton-Schulz5 orthogonalization matches Keller Jordan's spec.
def _zeropower_via_newtonschulz5(G, steps=8 ):
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


def muon_step(params, grads, m, v, step, lr, momentum=0.95, ns_steps=5, weight_decay=0.0):
    """Muon for 2D weights, AdamW for 1D params (LN gammas/biases, embeddings)."""
    for k in params:
        g = grads[k]
        if g.ndim >= 2:
            m[k] = momentum * m[k] + g
            update = _zeropower_via_newtonschulz5(m[k], steps=ns_steps)
            scale = 0.2 * math.sqrt(max(update.shape[0], update.shape[1]))
            params[k] -= lr * scale * update + lr * weight_decay * params[k]
        else:
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.95 * v[k] + 0.05 * (g * g)
            m_hat = m[k] / (1 - 0.9 ** step)
            v_hat = v[k] / (1 - 0.95 ** step)
            params[k] -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + weight_decay * params[k])

log = lambda msg: (print(msg), sys.stdout.flush())

log(f"Model: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}, ctx={CTX_LEN}")
log(f"Vocab: {repr(VOCAB[:20])}...{repr(VOCAB[-5:])}")

# Verify round-trip
test = "Hello world! What's 1+1?"
ids = encode(test)
back = ''.join(VOCAB[i] for i in ids if 0 <= i < len(VOCAB))
log(f"Round-trip: {repr(test)} -> {repr(back)}")
assert back == test, f"Round-trip failed: {repr(test)} != {repr(back)}"
log("Round-trip OK")

params = init_weights(seed=42)
total = param_size(params)
log(f"Total params: {total:,}")

# Load training data
with open('train/chat.txt', 'r', encoding='ascii') as f:
    text = f.read()
log(f"Data: {len(text):,} chars")

# Oversample test QA pairs
test_qa = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 9+10?\nBot:", "9+10 equals 19!"),
    ("Human: What's the capital of Germany?\nBot:", "Berlin is the capital of Germany."),
    ("Human: What's 3*3?\nBot:", "3*3 is 9")
]
# extra_parts = [p + a + "\n" for _ in range(500) for p, a in test_qa]
# extra = "".join(extra_parts)
# log(f"Extra oversampling text: {len(extra):,} chars")

# data = np.concatenate([encode(text), encode(extra)])
data = encode(text)
log(f"With oversampling: {len(data):,} tokens")

# Generation helper
def generate(params, prompt, max_new=100, temperature=1.0):
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        ctx = tokens[-CTX_LEN:]
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
        if next_id in [VOCAB.index('!'), VOCAB.index('.')]:
            break
    return ''.join(VOCAB[i] if 0 <= i < len(VOCAB) else '?' for i in tokens[len(encode(prompt)):])

# Train
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
STEPS, BS = 15000, 16
# Muon peak LR; effective update is lr * 0.2 * sqrt(max(rows, cols)) ~ lr * 2.7,
# so 1e-3 base ≈ equivalent to AdamW 3e-3 (more aggressive, but Muon is well-behaved).
MUON_LR = 1e-2

start = time.time()
for step in range(STEPS):
    x, y = get_batch(data, BS, CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(BS)[:, None], np.arange(CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % 500 == 0:
        elapsed = time.time() - start
        lr_now = get_lr(step, 200, STEPS, MUON_LR, MUON_LR * 0.1)
        log(f"step {step:5d} loss={loss:.4f} lr={lr_now:.6f} elapsed={elapsed:.0f}s")
        correct = 0
        for p, _ in test_qa:
            s = generate(params, p, max_new=50, temperature=0.0)
            log(f"  > {repr(s)}")
            if "Hello" in p and any(w in s.lower() for w in ["hello", "hi"]):
                correct += 1
            elif "9+10" in p and "19" in s:
                correct += 1
            elif "France" in p and "paris" in s.lower():
                correct += 1
            elif "3*3" in p and "9" in s:
                correct += 1
        log(f"  Correct: {correct}/4")
        if correct == 4:
            log("  *** ALL 4 CORRECT! ***")
            # break
    
    grads = backward_full(params, logits, y, cache)
    lr_now = get_lr(step, 200, STEPS, MUON_LR, MUON_LR * 0.1)
    muon_step(params, grads, m, v, step + 1, lr_now)

log(f"Training done. Final loss: {loss:.4f}")

# Export
os.makedirs('www', exist_ok=True)
weights_list = [params['emb'].tolist()]
for i in range(N_LAYERS):
    for key in (f'l{i}.ln1_g', f'l{i}.ln1_b', f'l{i}.wq', f'l{i}.wk',
                f'l{i}.wv', f'l{i}.wo', f'l{i}.ln2_g', f'l{i}.ln2_b',
                f'l{i}.w1', f'l{i}.b1', f'l{i}.w2', f'l{i}.b2'):
        weights_list.append(params[key].tolist())
weights_list.append(params['lnf_g'].tolist())
weights_list.append(params['lnf_b'].tolist())

out = {
    'config': {
        'vocab_size': VOCAB_SIZE,
        'd_model': D_MODEL,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'head_dim': HEAD_DIM,
        'd_ff': D_FF,
        'ctx_len': CTX_LEN,
    },
    'weights': weights_list,
}
with open('www/weights.json', 'w') as f:
    json.dump(out, f)
size_mb = os.path.getsize('www/weights.json') / 1e6
log(f"Exported weights.json: {size_mb:.1f} MB")
log(f"Weight arrays: {len(weights_list)}")
