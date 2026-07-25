"""Quick test: greedy decoding with correctly loaded weights from weights.json."""
import sys, json, os, numpy as np
sys.path.insert(0, 'train')

# Load weights from www/weights.json
with open('www/weights.json') as f:
    data = json.load(f)

# Build params dict matching train_short.py's export order
params = {}
names = ['emb']
for i in range(2):
    for k in ['ln1_g','ln1_b','wq','wk','wv','wo','ln2_g','ln2_b','w1','b1','w2','b2']:
        names.append(f'l{i}.{k}')
names.extend(['lnf_g', 'lnf_b'])

for i, name in enumerate(names):
    params[name] = np.array(data['weights'][i], dtype=np.float32)

print(f"Loaded {len(params)} params from weights.json")
for k, v in params.items():
    print(f"  {k}: {v.shape}")

# Config
V = data['config']['vocab_size']
D = data['config']['d_model']
N_HEADS = data['config']['n_heads']
HEAD_DIM = data['config']['head_dim']
N_LAYERS = data['config']['n_layers']
D_FF = data['config']['d_ff']

def encode(text):
    return [ord(c) - 32 if 0 <= ord(c) - 32 < V else 0 for c in text]

def decode(ids):
    return ''.join(chr(i+32) for i in ids if 0 <= i < V)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta

def attention(q, k, v):
    B, H, T, HD = q.shape
    scale = 1.0 / np.sqrt(HD)
    att = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
    att = np.tanh(att) * 0.5 + 0.5
    y = np.einsum('bhqk,bhvd->bhqd', att, v)
    return y

def forward_full(params, xb):
    B, T = xb.shape
    emb = params['emb']
    x = emb[xb]
    
    for li in range(N_LAYERS):
        ln1_g = params[f'l{li}.ln1_g']
        ln1_b = params[f'l{li}.ln1_b']
        wq = params[f'l{li}.wq']
        wk = params[f'l{li}.wk']
        wv = params[f'l{li}.wv']
        wo = params[f'l{li}.wo']
        ln2_g = params[f'l{li}.ln2_g']
        ln2_b = params[f'l{li}.ln2_b']
        w1 = params[f'l{li}.w1']
        b1 = params[f'l{li}.b1']
        w2 = params[f'l{li}.w2']
        b2 = params[f'l{li}.b2']
        
        x_norm = layernorm(x, ln1_g, ln1_b)
        q = x_norm @ wq.T
        k = x_norm @ wk.T
        v = x_norm @ wv.T
        q = q.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        y = attention(q, k, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D) @ wo.T
        x = x + y
        
        x_norm2 = layernorm(x, ln2_g, ln2_b)
        hidden = gelu(x_norm2 @ w1 + b1)
        f = hidden @ w2 + b2
        x = x + f
    
    x = layernorm(x, params['lnf_g'], params['lnf_b'])
    logits = x @ emb.T
    return logits, x

def generate(prompt, max_new=100, temperature=0.0):
    tokens = encode(prompt)
    for _ in range(max_new):
        ctx = tokens[-64:]
        xb = np.array([ctx], dtype=np.int32)
        logits, _ = forward_full(params, xb)
        last = logits[0, -1, :]
        if temperature > 0:
            last = last / temperature
            probs = np.exp(last - last.max())
            probs = probs / probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        else:
            next_id = int(np.argmax(last))
        tokens.append(next_id)
        if next_id == 0:
            break
    return decode(tokens[len(encode(prompt)):])

prompts = [
    ("Human: Hello\nBot:", "hello"),
    ("Human: What's 1+1?\nBot:", "2"),
    ("Human: What's the capital of France?\nBot:", "paris"),
]

correct = 0
for p, kw in prompts:
    s = generate(p, max_new=100, temperature=0.0)
    print(f"Q: {repr(p.split(chr(10))[0])}")
    print(f"A: {repr(s)}")
    hit = kw in s.lower()
    print(f"  {'PASS' if hit else 'FAIL'} (looking for '{kw}')")
    if hit:
        correct += 1

print(f"\nScore: {correct}/3")
