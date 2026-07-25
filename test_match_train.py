"""Test model inference - using model_95c directly (already has VOCAB_SIZE=95)."""
import sys, json, os, importlib
import numpy as np

sys.path.insert(0, 'train')

# Import directly - model.py already has VOCAB_SIZE=95
import model_95c as _m
importlib.reload(_m)
from model_95c import init_weights, forward_full, VOCAB_SIZE

print(f"Model VOCAB_SIZE = {VOCAB_SIZE}")

with open('www/weights.json') as f:
    data = json.load(f)

print(f"Weights.json vocab_size = {data['config']['vocab_size']}")

params = init_weights(seed=42)
names = ['emb']
for i in range(2):
    for k in ['ln1_g','ln1_b','wq','wk','wv','wo','ln2_g','ln2_b','w1','b1','w2','b2']:
        names.append(f'l{i}.{k}')
names.extend(['lnf_g', 'lnf_b'])
for i, n in enumerate(names):
    params[n] = np.array(data['weights'][i], dtype=np.float32)

print(f"Loaded params, emb shape: {params['emb'].shape}")

def encode(text):
    return np.array([max(0, min(VOCAB_SIZE - 1, ord(c) - 32)) for c in text], dtype=np.int32)

def decode(ids):
    return ''.join(chr(int(i) + 32) if 0 <= i < VOCAB_SIZE else '?' for i in ids)

def generate(params, prompt, max_new=100):
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        ctx = tokens[-64:]
        x = np.array([ctx], dtype=np.int32)
        logits, _ = forward_full(params, x)
        nid = int(logits[0, -1, :].argmax())
        tokens.append(nid)
        if nid == 0:
            break
    return decode(tokens[len(encode(prompt)):])

prompts = [
    ("Human: Hello\nBot:", "hi"),
    ("Human: What's 1+1?\nBot:", "2"),
    ("Human: What's the capital of France?\nBot:", "paris"),
]

correct = 0
for p, kw in prompts:
    s = generate(params, p, max_new=100)
    print(f"Q: {repr(p.split(chr(10))[0])}")
    print(f"A: {repr(s)}")
    hit = kw in s.lower()
    print(f"  {'PASS' if hit else 'FAIL'} (looking for '{kw}')")
    if hit:
        correct += 1

print(f"\nScore: {correct}/3")
