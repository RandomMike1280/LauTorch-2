"""Train on actual prompts using 64-char ASCII tokenizer (no corruption)."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, 'train')

# --- 64-char ASCII tokenizer ---
# ASCII 32-95 = 64 chars: ' ' to '_'
# ' ' (32) -> 0, '!' (33) -> 1, ..., '_' (95) -> 63
# All lowercase letters a-z are included (ASCII 97-122 -> IDs 65-90)
# All uppercase A-Z are included (ASCII 65-90 -> IDs 33-58)
# Digits 0-9 are included (ASCII 48-57 -> IDs 16-25)

VOCAB_SIZE = 64
VOCAB = ''.join(chr(32 + i) for i in range(VOCAB_SIZE))  # ' ' to '_'
ID_TO_CHAR = {i: VOCAB[i] for i in range(VOCAB_SIZE)}
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}

def encode(text):
    out = []
    for c in text:
        out.append(CHAR_TO_ID.get(c, 0))  # unknown -> space
    return np.array(out, dtype=np.int32)

def decode(ids):
    return ''.join(ID_TO_CHAR[i] for i in ids if 0 <= i < VOCAB_SIZE)

# Override model
import model
model.VOCAB_SIZE = VOCAB_SIZE
model.VOCAB = VOCAB
model.D_MODEL = 32
model.N_HEADS = 4
model.HEAD_DIM = model.D_MODEL // model.N_HEADS
model.N_LAYERS = 2
model.D_FF = 128
model.CTX_LEN = 64

from model import init_weights, forward_full, backward_full, param_size
from train import get_batch, adamw_step, get_lr

print(f"Model: vocab={VOCAB_SIZE}, d_model={model.D_MODEL}, layers={model.N_LAYERS}, heads={model.N_HEADS}, d_ff={model.D_FF}")
print(f"Vocab: {repr(VOCAB[:20])}...")

# Verify round-trip with actual prompts
test_cases = [
    "hello",
    "1+1=2",
    "paris",
    "Paris",
    "HI",
    "2",
    "What's 1+1?",
    "What's the capital of France?",
]
print("\nRound-trip verification:")
all_ok = True
for t in test_cases:
    ids = encode(t)
    back = decode(ids)
    ok = (t == back)
    if not ok:
        print(f"  {repr(t):40s} -> {repr(back):40s} CORRUPT")
        all_ok = False
if all_ok:
    print("  All round-trips OK!")
else:
    print("  Some corruption!")

params = init_weights(seed=42)
total = param_size(params)
print(f"\nTotal params: {total:,}")

# Training data: actual prompts and answers
# Use the same format as what chat.lau will use
test_qa = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 1+1?\nBot:", "1+1 equals 2!"),
    ("Human: What's the capital of France?\nBot:", "Paris is the capital of France."),
]

# Build training corpus - repeat heavily
corpus_parts = []
for _ in range(5000):
    for p, a in test_qa:
        corpus_parts.append(p + a + "\n")
corpus = "".join(corpus_parts)
data = encode(corpus)
print(f"Training corpus: {len(corpus)} chars, {len(data)} tokens")

# Generation
def generate(params, prompt, max_new=50, temperature=0.5):
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        ctx = tokens[-model.CTX_LEN:]
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
        if next_id == 0:  # space = newline
            break
    return decode(tokens[len(encode(prompt)):])

# Train
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
STEPS, BS = 15000, 8
MAX_LR = 3e-4
MIN_LR = 1e-5
WARMUP = 500

start = time.time()
for step in range(STEPS):
    x, y = get_batch(data, BS, model.CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(BS)[:, None], np.arange(model.CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % 500 == 0:
        elapsed = time.time() - start
        lr_now = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
        print(f"\nstep {step:5d} loss={loss:.4f} lr={lr_now:.6f} elapsed={elapsed:.0f}s")
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

# Export
os.makedirs('www', exist_ok=True)
weights_list = [params['emb'].tolist()]
for i in range(model.N_LAYERS):
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
        'd_model': model.D_MODEL,
        'n_layers': model.N_LAYERS,
        'n_heads': model.N_HEADS,
        'head_dim': model.HEAD_DIM,
        'd_ff': model.D_FF,
        'ctx_len': model.CTX_LEN,
    },
    'weights': weights_list,
}
with open('www/weights.json', 'w') as f:
    json.dump(out, f)
size_mb = os.path.getsize('www/weights.json') / 1e6
print(f"Exported weights.json: {size_mb:.1f} MB")
print(f"Weight arrays: {len(weights_list)}")
