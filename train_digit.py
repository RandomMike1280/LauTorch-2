"""Train on digit answers using tiny digit vocabulary."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, 'train')

# --- Digit tokenizer config (16 chars) ---
DIGIT_VOCAB = " 0123456789.\n"
DIGIT_VOCAB_SIZE = len(DIGIT_VOCAB)  # 16
DIGIT_ID_TO_CHAR = {i: c for i, c in enumerate(DIGIT_VOCAB)}
DIGIT_CHAR_TO_ID = {c: i for i, c in enumerate(DIGIT_VOCAB)}

def encode(text):
    out = []
    for c in text:
        out.append(DIGIT_CHAR_TO_ID.get(c, 0))
    return np.array(out, dtype=np.int32)

def decode(ids):
    return ''.join(DIGIT_ID_TO_CHAR[i] for i in ids)

# --- Override model.py BEFORE importing ---
import model
model.VOCAB_SIZE = DIGIT_VOCAB_SIZE
model.VOCAB = DIGIT_VOCAB
model.D_MODEL = 24
model.N_HEADS = 4
model.HEAD_DIM = model.D_MODEL // model.N_HEADS
model.N_LAYERS = 2
model.D_FF = 96
model.CTX_LEN = 64

from model import init_weights, forward_full, backward_full, param_size
from train import get_batch, adamw_step, get_lr

print(f"Model: vocab={DIGIT_VOCAB_SIZE}, d_model={model.D_MODEL}, layers={model.N_LAYERS}, heads={model.N_HEADS}, d_ff={model.D_FF}")
print(f"Vocab: {repr(DIGIT_VOCAB)}")

# Verify round-trip
test = " 1+1=2\n"
ids = encode(test)
back = decode(ids)
print(f"Round-trip: {repr(test)} -> {repr(back)} -> Match: {test == back}")

params = init_weights(seed=42)
total = param_size(params)
print(f"Total params: {total:,}")

# Training data: digit answers only
# Format: "HELLO 10" (question + space + answer + newline)
# The model predicts answer tokens given question tokens
test_qa = [
    ("HELLO", "10"),       # Hello -> answer "10"
    ("1+1=2", "20"),      # 1+1=2 -> answer "20"
    ("PARIS", "30"),       # Paris -> answer "30"
]

# Eval prompts - SAME format as training
eval_prompts = [
    ("HELLO ", "10"),       # expects "10" in output
    ("1+1=2 ", "20"),      # expects "20"
    ("PARIS ", "30"),       # expects "30"
]

# Build training corpus
training_text = ""
for _ in range(3000):
    for q, a in test_qa:
        training_text += q + " " + a + "\n"

data = encode(training_text)
print(f"Training corpus: {len(training_text)} chars, {len(data)} tokens")

# Generation
def generate(params, prompt, max_new=10, temperature=0.5):
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
        if next_id == 0:  # space = stop
            break
        if next_id == 14:  # newline = stop
            break
    return decode(tokens[len(encode(prompt)):])

# Train
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
STEPS, BS = 10000, 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 200

start = time.time()
for step in range(STEPS):
    x, y = get_batch(data, BS, model.CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(BS)[:, None], np.arange(model.CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % 200 == 0:
        elapsed = time.time() - start
        lr_now = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
        print(f"step {step:5d} loss={loss:.4f} lr={lr_now:.6f} elapsed={elapsed:.0f}s")
        correct = 0
        for p, expected in eval_prompts:
            s = generate(params, p, max_new=10, temperature=0.5)
            hit = expected in s
            print(f"  Q: {repr(p[:30])} | A: {repr(s)} | Exp: {repr(expected)} | {'PASS' if hit else 'FAIL'}")
            if hit:
                correct += 1
        print(f"  Correct: {correct}/3")
        if correct == 3:
            print("  *** ALL 3 CORRECT! ***")
            break
    
    grads = backward_full(params, logits, y, cache)
    lr_now = get_lr(step, WARMUP, STEPS, MAX_LR, MIN_LR)
    adamw_step(params, grads, m, v, step + 1, lr_now)

print(f"Training done. Final loss: {loss:.4f}")

# Export to JSON
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
        'vocab_size': DIGIT_VOCAB_SIZE,
        'vocab': DIGIT_VOCAB,
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
