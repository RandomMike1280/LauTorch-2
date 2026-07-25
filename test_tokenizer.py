"""Test the trained model by running the same inference as chat.lau would do."""
import json
import math

# Load weights
with open('www/weights.json') as f:
    data = json.load(f)

config = data['config']
weights = data['weights']
VOCAB = config['vocab']
V = config['vocab_size']
D = config['d_model']
H = config['n_heads']
HD = config['head_dim']
FF = config['d_ff']
CTX = config['ctx_len']

print(f"Config: V={V}, D={D}, H={H}, HD={HD}, FF={FF}, CTX={CTX}")
print(f"Vocab: {repr(VOCAB[:20])}...")

# Verify weight dimensions
emb = weights[0]  # W[1] - embedding (V x D)
print(f"Embedding shape: {len(emb)} rows x {len(emb[0]) if emb else 'N/A'} cols (expected {V} x {D})")

# Tokenizer
def encode(s):
    out = []
    for i in range(len(s)):
        c = s[i]
        p = VOCAB.find(c)
        if p == -1:
            p = 0
        out.append(p)
    return out

def decode(t):
    return ''.join(VOCAB[i] if i < len(VOCAB) else '?' for i in t)

# Test encode/decode
test_str = "Hello"
tokens = encode(test_str)
print(f"Encode '{test_str}': {tokens}")
print(f"Decode {tokens}: {decode(tokens)}")

# Check lowercase
lower_str = "hello"
tokens_lower = encode(lower_str)
print(f"Encode '{lower_str}': {tokens_lower}")
print(f"Decode {tokens_lower}: {decode(tokens_lower)}")

# Check question with apostrophe
q_str = "What's 1+1?"
tokens_q = encode(q_str)
print(f"Encode '{q_str}': {tokens_q}")
print(f"Decode {tokens_q}: {decode(tokens_q)}")

# Simple test: embed lookup
print("\nEmbedding lookup test:")
for tid in [0, 1, 2, 3, 4, 5]:  # space, !, ", #, $, %
    vec = emb[tid]
    print(f"  ID {tid} ({repr(VOCAB[tid])}): first 5 dims = {vec[:5]}")

print("\nIf decode/encode round-trips correctly, the tokenizer is working.")
print("The issue with underscores in the previous run was likely from tokenization.")
