"""Train with SHORT answers - easier to generate."""
import os, sys, time, json, importlib
import numpy as np

VOCAB_SIZE = 95
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))

def encode(text):
    ids = []
    for c in text:
        oid = ord(c) - 32
        ids.append(max(0, min(VOCAB_SIZE - 1, oid)))
    return np.array(ids, dtype=np.int32)

def decode(ids):
    chars = []
    for i in ids:
        idx = int(i)
        if 0 <= idx < VOCAB_SIZE:
            chars.append(chr(idx + 32))
        else:
            chars.append('?')
    return ''.join(chars)

# Patch model.py
with open('train/model.py') as f:
    src = f.read()
old_block = "VOCAB_SIZE = 95\nVOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'"
new_block = f"VOCAB_SIZE = {VOCAB_SIZE}\nVOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'"
src = src.replace(old_block, new_block)
with open('train/model_95b.py', 'w') as f:
    f.write(src)

sys.path.insert(0, 'train')
import model_95b as _m
importlib.reload(_m)

from model_95b import init_weights, forward_full, backward_full, param_size
from model_95b import VOCAB_SIZE as VS, VOCAB as V, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN

print(f"Model: vocab={VS}, d_model={D_MODEL}, layers={N_LAYERS}")
sys.stdout.flush()

for t in ["hello", "1+1=2", "Paris", "HI", "2"]:
    ids = encode(t)
    back = decode(ids)
    assert t == back, f"FAIL: {t} -> {back}"
print("Round-trip OK")
sys.stdout.flush()

params = init_weights(seed=42)
print(f"Params: {param_size(params):,}")
sys.stdout.flush()

# SHORT answers
test_qa = [
    ("Human: Hello\nBot:", "Hi"),
    ("Human: What's 1+1?\nBot:", "2"),
    ("Human: What's the capital of France?\nBot:", "Paris"),
]

corpus = ""
for _ in range(5000):
    for p, a in test_qa:
        corpus += p + a + "\n"

data = encode(corpus)
print(f"Corpus: {len(corpus)} chars, {len(data)} tokens")
sys.stdout.flush()

def generate(params, prompt, max_new=20, temperature=0.5):
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        ctx = tokens[-CTX_LEN:]
        x = np.array([ctx], dtype=np.int32)
        logits, _ = forward_full(params, x)
        last = logits[0, -1, :]
        if temperature <= 0:
            nid = int(last.argmax())
        else:
            s = last - last.max()
            p = np.exp(s) / np.exp(s).sum()
            nid = int(np.random.choice(len(p), p=p))
        tokens.append(nid)
        if nid == 0:
            break
    return decode(tokens[len(encode(prompt)):])

def adamw(params, grads, m, v, step, lr):
    for k in params:
        g = grads[k]
        m[k] = 0.9 * m[k] + 0.1 * g
        v[k] = 0.999 * v[k] + 0.001 * g * g
        m_hat = m[k] / (1 - 0.9 ** step)
        v_hat = v[k] / (1 - 0.999 ** step)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * (step - warmup) / (max_steps - warmup)))

def get_batch(data, bs, ctx):
    starts = np.random.randint(0, len(data) - ctx - 1, bs)
    x = np.stack([data[s:s + ctx] for s in starts])
    y = np.stack([data[s + 1:s + ctx + 1] for s in starts])
    return x.astype(np.int32), y.astype(np.int32)

m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}

STEPS, BS = 15000, 8
MAX_LR, MIN_LR, WARMUP = 3e-4, 1e-5, 500

start = time.time()
for step in range(STEPS):
    x, y = get_batch(data, BS, CTX_LEN)
    logits, cache = forward_full(params, x)
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    loss = -logp[np.arange(BS)[:, None], np.arange(CTX_LEN)[None, :], y].mean()

    if step % 200 == 0:
        elapsed = time.time() - start
        lr = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
        print(f"step {step:5d} loss={loss:.4f} lr={lr:.6f} elapsed={elapsed:.0f}s")
        sys.stdout.flush()
        correct = 0
        for p, expected in test_qa:
            s = generate(params, p, max_new=20, temperature=0.5)
            hit = expected.lower() in s.lower()
            print(f"  {repr(p.split(chr(10))[0])} -> {repr(s)} {'PASS' if hit else 'FAIL (exp ' + expected + ')'}")
            if hit:
                correct += 1
        print(f"  Score: {correct}/3")
        sys.stdout.flush()
        if correct == 3:
            print("*** ALL 3! ***")
            break

    grads = backward_full(params, logits, y, cache)
    lr = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
    adamw(params, grads, m, v, step + 1, lr)

print(f"Done. Loss: {loss:.4f}")
sys.stdout.flush()

# Export
os.makedirs('www', exist_ok=True)
weights_list = [params['emb'].tolist()]
for i in range(N_LAYERS):
    for k in [f'l{i}.ln1_g', f'l{i}.ln1_b', f'l{i}.wq', f'l{i}.wk',
               f'l{i}.wv', f'l{i}.wo', f'l{i}.ln2_g', f'l{i}.ln2_b',
               f'l{i}.w1', f'l{i}.b1', f'l{i}.w2', f'l{i}.b2']:
        weights_list.append(params[k].tolist())
weights_list.append(params['lnf_g'].tolist())
weights_list.append(params['lnf_b'].tolist())

out = {'config': {'vocab_size': VS, 'vocab': V, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
        'n_heads': N_HEADS, 'head_dim': HEAD_DIM, 'd_ff': D_FF, 'ctx_len': CTX_LEN},
       'weights': weights_list}
with open('www/weights.json', 'w') as f:
    json.dump(out, f)
print(f"Exported: {os.path.getsize('www/weights.json') / 1e6:.1f} MB")
