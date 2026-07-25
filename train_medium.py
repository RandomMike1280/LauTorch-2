"""Train the medium model (d_model=32, 2 layers)."""
import os, sys, time, json
sys.path.insert(0, 'train')
import numpy as np
from model import (init_weights, forward_full, backward_full, param_size,
                   VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN)
from train import encode, get_batch, adamw_step, get_lr, generate

print(f"Model: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
params = init_weights(seed=42)
print(f"Total params: {param_size(params)}")

# Load chat data
with open('train/chat.txt', 'r', encoding='ascii') as f:
    text = f.read()
print(f"Training data: {len(text)} chars")
data = encode(text)

# Oversample test prompts 500x
test_answers = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 1+1?\nBot:", "1+1 equals 2!"),
    ("Human: What's the capital of France?\nBot:", "Paris is the capital of France."),
]
extra = ""
for _ in range(500):
    for p, a in test_answers:
        extra += p + a + "\n"
data_extra = encode(extra)
data = np.concatenate([data, data_extra])
print(f"With oversampling: {len(data)} tokens ({len(data)/1000:.0f}K)")

# Train
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}
steps = 20000
batch_size = 16
os.makedirs('train/weights', exist_ok=True)

start = time.time()
for step in range(steps):
    x, y = get_batch(data, batch_size, CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(batch_size)[:, None], np.arange(CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % 500 == 0:
        elapsed = time.time() - start
        lr = get_lr(step, 200, steps, 3e-4, 3e-5)
        print(f"step {step:5d} loss={loss:.4f} lr={lr:.6f} elapsed={elapsed:.0f}s")
        correct = 0
        for p, _ in test_answers:
            s = generate(params, p, max_new=30, temperature=0.3)
            print(f"  > {repr(s)}")
            if "Hello" in p and any(w in s.lower() for w in ["hello", "hi"]):
                correct += 1
            elif "1+1" in p and "2" in s:
                correct += 1
            elif "Paris" in p and "paris" in s.lower():
                correct += 1
        print(f"  Correct: {correct}/3")
        if correct == 3:
            print("  *** ALL 3 CORRECT! ***")
        
    if step > 0 and step % 5000 == 0:
        np.savez(f'train/weights/model_medium_{step}.npz', **{k: v for k, v in params.items()})
    
    grads = backward_full(params, logits, y, cache)
    lr = get_lr(step, 200, steps, 3e-4, 3e-5)
    adamw_step(params, grads, m, v, step + 1, lr)

print(f"\nTraining complete. Final loss: {loss:.4f}")
print("Final samples:")
for p, _ in test_answers:
    s = generate(params, p, max_new=40, temperature=0.3)
    print(f"  > {repr(s)}")

np.savez('train/weights/model_medium.npz', **{k: v for k, v in params.items()})
print("Saved to train/weights/model_medium.npz")
