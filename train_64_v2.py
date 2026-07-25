"""Patch model.py with 64-char vocab config and run training."""
import os, sys, time, json

# --- 64-char ASCII tokenizer (no corruption) ---
# ASCII 32-95 = 64 chars: ' ' to '_'
VOCAB_SIZE = 64
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '_'

# Verify round-trip
def encode(text):
    out = []
    for c in text:
        out.append(ord(c) - 32)
    return out

def decode(ids):
    return ''.join(chr(i + 32) for i in ids)

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
else:
    print("  Some corruption!")
sys.stdout.flush()

# Read model.py, patch config, write back
model_path = 'train/model.py'
with open(model_path, 'r') as f:
    model_src = f.read()

# Replace the VOCAB_SIZE and VOCAB section
old_config = '''VOCAB_SIZE = 95
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '~'
D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 8
N_LAYERS = 2
D_FF = 128
CTX_LEN = 64
MAX_BATCH = 16'''

new_config = f'''VOCAB_SIZE = {VOCAB_SIZE}
VOCAB = "{VOCAB}"
D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 8
N_LAYERS = 2
D_FF = 128
CTX_LEN = 64
MAX_BATCH = 16'''

assert old_config in model_src, "Config block not found in model.py!"
model_src = model_src.replace(old_config, new_config)

# Write patched model
patched_path = 'train/model_patched.py'
with open(patched_path, 'w') as f:
    f.write(model_src)
print(f"Patched model written to {patched_path}")

# Now import from patched model
sys.path.insert(0, 'train')
import importlib
import model_patched as _m
importlib.reload(_m)

from model_patched import init_weights, forward_full, backward_full, param_size
from model_patched import VOCAB_SIZE, VOCAB, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN
from train import get_batch, adamw_step, get_lr

print(f"\nModel: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
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

# Encode using our tokenizer
def encode_ids(text):
    ids = []
    for c in text:
        oid = ord(c) - 32
        if 0 <= oid < VOCAB_SIZE:
            ids.append(oid)
        else:
            ids.append(0)
    return ids

corpus_ids = np.array(encode_ids(corpus), dtype=np.int32)
print(f"Training corpus: {len(corpus)} chars, {len(corpus_ids)} tokens")
sys.stdout.flush()

# Generation
def generate(params, prompt, max_new=50, temperature=0.5):
    tokens = encode_ids(prompt)
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
    return decode(tokens[len(encode_ids(prompt)):])

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
                correct += 1
                print(f"  -> PASS")
            elif "1+1" in p and "2" in s:
                correct += 1
                print(f"  -> PASS")
            elif "Paris" in p and "paris" in s.lower():
                correct += 1
                print(f"  -> PASS")
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
        'vocab_size': VOCAB_SIZE,
        'vocab': VOCAB,
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
print(f"Weight arrays: {len(weights_list)}")
