"""Test the large model on 3 prompts."""
import os, sys, json
sys.path.insert(0, 'train')

# First check model architecture
from model import VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, param_size, init_weights, forward_full
print(f"Model: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, d_ff={D_FF}")
print(f"Expected params: {param_size(init_weights())}")

# Load weights
np_w = __import__('numpy').load('train/weights/model.npz')
w = {k: np_w[k] for k in np_w.files}
print(f"Loaded weights: {list(w.keys())}")
actual_params = sum(v.size for v in w.values())
print(f"Loaded param count: {actual_params}")
expected = param_size(init_weights())
print(f"Expected param count: {expected}")
if actual_params != expected:
    print("MISMATCH! Architecture may have changed.")
    sys.exit(1)

# Test generate
from train import encode, decode, generate

test_prompts = [
    "Human: Hello\nBot:",
    "Human: What's 1+1?\nBot:",
    "Human: What's the capital of France?\nBot:",
]

print("\n=== Test Results ===")
correct = 0
for p in test_prompts:
    out = generate(w, p, max_new=40, temperature=0.0)  # greedy
    print(f"Q: {repr(p.split('Bot:')[0].strip())}")
    print(f"A: {repr(out)}")
    # Check
    if "Hello" in p and ("hello" in out.lower() or "hi" in out.lower()):
        correct += 1
        print("PASS")
    elif "1+1" in p and ("2" in out):
        correct += 1
        print("PASS")
    elif "capital of France" in p and ("paris" in out.lower()):
        correct += 1
        print("PASS")
    else:
        print("FAIL")
    print()
print(f"Score: {correct}/3")
