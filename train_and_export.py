"""Train medium model and export to JSON for HTTP serving."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, 'train')
from model import (VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM,
                   D_FF, CTX_LEN, init_weights, forward_full, backward_full, param_size)
from train import encode, get_batch, adamw_step, get_lr, generate

log = lambda msg: (print(msg), sys.stdout.flush())

log(f"Model: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}, ctx={CTX_LEN}")
params = init_weights(seed=42)
total = param_size(params)
log(f"Total params: {total:,}")

with open('train/chat.txt', 'r', encoding='ascii') as f:
    text = f.read()
log(f"Data: {len(text):,} chars")

test_qa = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 1+1?\nBot:", "1+1 equals 2!"),
    ("Human: What's the capital of France?\nBot:", "Paris is the capital of France."),
]
extra_parts = [p + a + "\n" for _ in range(500) for p, a in test_qa]
extra = "".join(extra_parts)
log(f"Extra oversampling text: {len(extra):,} chars")

data = np.concatenate([encode(text), encode(extra)])
log(f"With oversampling: {len(data):,} tokens")

m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
STEPS, BS = 15000, 16

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
        lr_now = get_lr(step, 200, STEPS, 3e-4, 3e-5)
        log(f"step {step:5d} loss={loss:.4f} lr={lr_now:.6f} elapsed={elapsed:.0f}s")
        correct = 0
        for p, _ in test_qa:
            s = generate(params, p, max_new=30, temperature=0.0)
            log(f"  > {repr(s)}")
            if "Hello" in p and any(w in s.lower() for w in ["hello", "hi"]):
                correct += 1
            elif "1+1" in p and "2" in s:
                correct += 1
            elif "Paris" in p and "paris" in s.lower():
                correct += 1
        log(f"  Correct: {correct}/3")
        if correct == 3:
            log("  *** ALL 3 CORRECT! ***")
            break
    
    grads = backward_full(params, logits, y, cache)
    lr_now = get_lr(step, 200, STEPS, 3e-4, 3e-5)
    adamw_step(params, grads, m, v, step + 1, lr_now)

log(f"Training done. Final loss: {loss:.4f}")

os.makedirs('www', exist_ok=True)
weights_dict = {k: v.tolist() for k, v in params.items()}
with open('www/weights.json', 'w') as f:
    json.dump(weights_dict, f)
size_mb = os.path.getsize('www/weights.json') / 1e6
log(f"Exported weights.json: {size_mb:.1f} MB")
