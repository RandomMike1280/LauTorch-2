"""Train with FULL 95-char ASCII tokenizer (space to tilde, all lowercase included)."""
import os, sys, time, json, importlib
import numpy as np

# --- 95-char printable ASCII (no corruption) ---
# ASCII 32-126 = 95 chars: ' ' to '~'
# All lowercase, uppercase, digits, punctuation included!
# We use chr(32) to chr(126) as the vocab
VOCAB_SIZE = 95
# Build vocab programmatically to avoid quote issues
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'

# Tokenizer
def encode(text):
    ids = []
    for c in text:
        oid = ord(c) - 32
        if 0 <= oid < VOCAB_SIZE:
            ids.append(oid)
        else:
            ids.append(0)  # clip to space
    return ids

def decode(ids):
    return ''.join(chr(i + 32) for i in ids if 0 <= i < VOCAB_SIZE)

# Verify round-trip
test_cases = ["hello", "1+1=2", "paris", "Paris", "HI", "2", "What's 1+1?"]
print("Round-trip verification:")
all_ok = True
for t in test_cases:
    ids = encode(t)
    back = decode(ids)
    ok = (t == back)
    if not ok:
        print(f"  {repr(t):30s} -> {repr(back):30s} CORRUPT")
        all_ok = False
if all_ok:
    print("  All round-trips OK!")
sys.stdout.flush()

# Patch model.py
model_path = 'train/model.py'
with open(model_path, 'r') as f:
    model_src = f.read()

old_block = """VOCAB_SIZE = 95
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'"""
# Build VOCAB as a Python list expression
vocab_list = "[" + ",".join(f"chr({32+i})" for i in range(VOCAB_SIZE)) + "]"
new_block = f"""VOCAB_SIZE = {VOCAB_SIZE}
VOCAB = ''.join({vocab_list})"""

assert old_block in model_src, "Could not find old block!"
model_src = model_src.replace(old_block, new_block)

patched_path = 'train/model_95.py'
with open(patched_path, 'w') as f:
    f.write(model_src)
print(f"Patched model written to {patched_path}")
sys.stdout.flush()

# Import patched model
sys.path.insert(0, 'train')
import model_95 as _m
importlib.reload(_m)

from model_95 import init_weights, forward_full, backward_full, param_size
from model_95 import VOCAB_SIZE as VS, VOCAB as V, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN

print(f"Model: vocab={VS}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}, ctx={CTX_LEN}")
assert VS == 95, f"Model VOCAB_SIZE is {VS}, not 95!"
assert len(V) == 95, f"Model VOCAB length is {len(V)}, not 95!"
sys.stdout.flush()

params = init_weights(seed=42)
total = param_size(params)
print(f"Total params: {total:,}")
sys.stdout.flush()

# Training data
test_qa = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 1+1?\nBot:", "1+1 equals 2!"),
    ("Human: What's the capital of France?\nBot:", "Paris is the capital of France."),
]

corpus_parts = []
for _ in range(5000):
    for p, a in test_qa:
        corpus_parts.append(p + a + "\n")
corpus = "".join(corpus_parts)
corpus_ids = np.array(encode(corpus), dtype=np.int32)
print(f"Training corpus: {len(corpus)} chars, {len(corpus_ids)} tokens")
sys.stdout.flush()

# Generation
def generate(params, prompt, max_new=50, temperature=0.5):
    tokens = encode(prompt)
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
        tokens.append(int(next_id))
        if next_id == 0:  # space = newline
            break
    return decode(tokens[len(encode(prompt)):])

# AdamW
def adamw_step(params, grads, m, v, step, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
        m_hat = m[k] / (1 - beta1 ** step)
        v_hat = v[k] / (1 - beta2 ** step)
        params[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * params[k])

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

def get_batch(data, batch_size, ctx_len):
    max_start = len(data) - ctx_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s:s + ctx_len] for s in starts])
    y = np.stack([data[s + 1:s + ctx_len + 1] for s in starts])
    return x.astype(np.int32), y.astype(np.int32)

# Train
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
STEPS, BS = 15000, 8
MAX_LR = 3e-4
MIN_LR = 1e-5
WARMUP = 500

start = time.time()
for step in range(STEPS):
    x, y = get_batch(corpus_ids, BS, CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(BS)[:, None], np.arange(CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % 500 == 0:
        elapsed = time.time() - start
        lr_now = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
        print(f"\nstep {step:5d} loss={loss:.4f} lr={lr_now:.6f} elapsed={elapsed:.0f}s")
        sys.stdout.flush()
        correct = 0
        for p, _ in test_qa:
            s = generate(params, p, max_new=40, temperature=0.5)
            print(f"  Q: {repr(p.split(chr(10))[0])}")
            print(f"  A: {repr(s)}")
            if "Hello" in p and any(w in s.lower() for w in ["hello", "hi"]):
                correct += 1; print(f"  -> PASS")
            elif "1+1" in p and "2" in s:
                correct += 1; print(f"  -> PASS")
            elif "Paris" in p and "paris" in s.lower():
                correct += 1; print(f"  -> PASS")
            else:
                print(f"  -> FAIL")
        print(f"Correct: {correct}/3")
        if correct == 3:
            print("*** ALL 3 CORRECT! ***")
            break
    
    grads = backward_full(params, logits, y, cache)
    lr_now = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
    adamw_step(params, grads, m, v, step + 1, lr_now)

print(f"\nTraining done. Final loss: {loss:.4f}")
sys.stdout.flush()

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
        'vocab_size': VS,
        'vocab': V,
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
print(f"Exported weights.json: {size_mb:.1f} MB")
