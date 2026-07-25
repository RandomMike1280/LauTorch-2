import re, sys
import numpy as np
sys.path.insert(0, 'train')

# Load all weights
def parse_laum(path):
    with open(path) as f:
        c = f.read()
    i = c.find('d="') + 3
    data = []
    while i < len(c):
        ch = c[i]
        if ch == '\\':
            i += 1
            data.append(c[i])
        elif ch == '"':
            break
        else:
            data.append(ch)
        i += 1
    data = ''.join(data)
    scales = [float(s) for s in re.search(r'c=\{([^}]+)\}', c).group(1).split(',')]
    shapes = [int(s) for s in re.search(r's=\{\s*([0-9,]+)\s*\}', c).group(1).split(',')]
    return data, scales, shapes

from quantize import _build_charmap
charmap = _build_charmap()

def dec(d, i):
    p = i * 2
    x = charmap.index(d[p])
    y = charmap.index(d[p+1])
    return (x * 95 + y) - 127

def get_layer_weights(path):
    data, scales, shapes = parse_laum(path)
    tensor_sizes = [shapes[i*2] * shapes[i*2+1] for i in range(len(shapes)//2)]
    offsets = [0]
    for s in tensor_sizes:
        offsets.append(offsets[-1] + s)
    def get(idx):
        return np.array([dec(data, offsets[idx]+j) for j in range(tensor_sizes[idx])]) * scales[idx]
    return tensor_sizes, get

emb_data, emb_scales, _ = parse_laum(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_emb.laum')
def dec_emb(i):
    p = i * 2
    x = charmap.index(emb_data[p])
    y = charmap.index(emb_data[p+1])
    return (x * 95 + y) - 127

emb_size = 96 * 16
embs = np.array([dec_emb(j) for j in range(emb_size)]).reshape(96, 16) * emb_scales[0]
print(f"embs[86] (1..8): {embs[86][:8]}")

l0_sizes, l0_get = get_layer_weights(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_l0.laum')
l1_sizes, l1_get = get_layer_weights(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_l1.laum')

# Build params dict (simplified)
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean()
    var = x.var()
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta

# Process 16 tokens
toks = [85, 77, 65, 78, 26, 0, 40, 69, 76, 76, 79, 0, 34, 79, 84, 26]
KV_cache_l0 = {'K': [], 'V': []}
KV_cache_l1 = {'K': [], 'V': []}

def forward_layer(x, get, KV):
    # LN1
    ln1_g = get(0)
    ln1_b = get(1)
    x_norm = layernorm(x, ln1_g, ln1_b)

    # Q, K, V
    wq = get(2).reshape(16, 16)
    wk = get(3).reshape(16, 16)
    wv = get(4).reshape(16, 16)
    Q = x_norm @ wq
    K = x_norm @ wk
    V = x_norm @ wv
    KV['K'].append(K)
    KV['V'].append(V)

    # Attention (single head at a time, then concat)
    P = len(KV['K'])
    attn_out = np.zeros(16)
    for hh in range(4):
        hs = hh * 4
        scores = np.zeros(P)
        for i in range(P):
            scores[i] = (Q[hs:hs+4] * KV['K'][i][hs:hs+4]).sum() * 0.5
        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()
        for d in range(4):
            s = 0
            for i in range(P):
                s += scores[i] * KV['V'][i][hs+d]
            attn_out[hs+d] = s

    # Output proj
    wo = get(5).reshape(16, 16)
    attn_proj = attn_out @ wo
    x1 = x + attn_proj

    # LN2
    ln2_g = get(6)
    ln2_b = get(7)
    x_norm2 = layernorm(x1, ln2_g, ln2_b)

    # MLP
    w1 = get(8).reshape(16, 16)
    b1 = get(9)
    w2 = get(10).reshape(16, 16)
    b2 = get(11)
    h_pre = x_norm2 @ w1 + b1
    h_act = gelu(h_pre)
    mlp_out = h_act @ w2 + b2

    return x1 + mlp_out

x = np.zeros(16)
KV_cache_l0 = {'K': [], 'V': []}
KV_cache_l1 = {'K': [], 'V': []}
for ti, tok in enumerate(toks):
    # Use 0-indexed: embs[tok] for token id tok (since embs[86] = 1-indexed token 86 = 0-indexed token 85 = "V")
    # Actually let me just use tok directly since toks contains 0-indexed ids (per my Lau's er(tok+1))
    # Lau er(tok+1) = (tok+1-1)*16+j = tok*16+j, reading row index tok (0-indexed)
    # So embs[tok] is the right row
    x = embs[tok]
    if ti == 0:
        print(f"emb (1..8): {x[:8]}")
    x = forward_layer(x, l0_get, KV_cache_l0)
    if ti == 0:
        print(f"Python after layer 0 token 1 (1..8): {x[:8]}")
    if ti == 7:
        print(f"Python after layer 0 token 8 (1..4): {x[:4]}")
    if ti == 15:
        print(f"Python after layer 0 token 16 (1..8): {x[:8]}")
    x = forward_layer(x, l1_get, KV_cache_l1)

print(f"Python layer 1 output token 16 (1..8): {x[:8]}")

# Apply final LN
lnf_g_arr = l1_get(12)  # tensor 13 in 1-indexed
lnf_b_arr = l1_get(13)  # tensor 14
x_norm = (x - x.mean()) / np.sqrt(x.var() + 1e-3)
x_final = x_norm * lnf_g_arr + lnf_b_arr
print(f"Python after final LN (1..8): {x_final[:8]}")

# Now do final LN and LM head