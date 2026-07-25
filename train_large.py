"""Train the large model (d_model=64, 3 layers) on chat data."""
import os, sys, time, json
sys.path.insert(0, 'train')
import numpy as np
from model import (init_weights, forward_full, backward_full, param_size,
                   VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN)
from train import encode, decode, get_batch, adamw_step, get_lr, generate

print(f"Model: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
print(f"Total params: {param_size(init_weights())}")

# Load chat data
with open('train/chat.txt', 'r', encoding='ascii') as f:
    text = f.read()
print(f"Loaded {len(text)} chars of training data")
data = encode(text)
print(f"Encoded {len(data)} tokens, range: {data.min()}..{data.max()}")

# Oversample test prompts heavily
test_answers = [
    ("Human: Hello\nBot:", "Hello! How are you?"),
    ("Human: What's 1+1?\nBot:", "1+1 equals 2!"),
    ("Human: What's the capital of France?\nBot:", "Paris is the capital of France."),
]
# Repeat each test QA pair 500x
extra_text = ""
for _ in range(500):
    for prompt, answer in test_answers:
        extra_text += prompt + answer + "\n"
# Append to training data
data_extra = encode(extra_text)
data = np.concatenate([data, data_extra])
print(f"With oversampling: {len(data)} tokens")

# Train
params = init_weights(seed=42)
m = {k: np.zeros_like(p) for k, p in params.items()}
v = {k: np.zeros_like(p) for k, p in params.items()}

steps = 30000
batch_size = 8
log_every = 500
save_every = 5000

os.makedirs('train/weights', exist_ok=True)

start = time.time()
for step in range(steps):
    x, y = get_batch(data, batch_size, CTX_LEN)
    logits, cache = forward_full(params, x)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(batch_size)[:, None], np.arange(CTX_LEN)[None, :], y]
    loss = nll.mean()
    
    if step % log_every == 0:
        elapsed = time.time() - start
        lr = get_lr(step, 200, steps, 3e-4, 3e-5)
        print(f"step {step:5d} loss={loss:.4f} lr={lr:.6f} elapsed={elapsed:.1f}s")
        
        # Sample test prompts
        samples = []
        for p, _ in test_answers:
            s = generate(params, p, max_new=30, temperature=0.3)
            samples.append(s)
            print(f"  > {repr(s)}")
        
        # Check correctness
        correct = 0
        for (_, expected_answer), out in zip(test_answers, samples):
            if "Hello" in expected_answer and ("hello" in out.lower() or "hi" in out.lower()):
                correct += 1
            elif "1+1" in expected_answer and "2" in out:
                correct += 1
            elif "Paris" in expected_answer and "paris" in out.lower():
                correct += 1
        print(f"  Correct: {correct}/3")
    
    if step > 0 and step % save_every == 0:
        path = f'train/weights/checkpoint_{step:06d}.npz'
        np.savez(path, **{k: v for k, v in params.items()})
        print(f"  Saved to {path}")
    
    grads = backward_full(params, logits, y, cache)
    lr = get_lr(step, 200, steps, 3e-4, 3e-5)
    adamw_step(params, grads, m, v, step + 1, lr)

print(f"\nTraining complete. Final loss: {loss:.4f}")
print("Final samples:")
for p, _ in test_answers:
    s = generate(params, p, max_new=40, temperature=0.3)
    print(f"  > {repr(s)}")

# Save final
np.savez('train/weights/model_large.npz', **{k: v for k, v in params.items()})
print("Saved to train/weights/model_large.npz")
