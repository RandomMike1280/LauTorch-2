"""Decode embedding from laum and compute token 1 layer 0 output, compare to Lau."""
import numpy as np
import sys, re
sys.path.insert(0, 'train')
from quantize import _build_charmap

charmap = _build_charmap()

def parse_laum(path):
    with open(path) as f:
        c = f.read()
    # Parse d="..." handling escapes
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

def dec(data, i):
    p = i * 2
    c1 = data[p]
    c2 = data[p+1]
    x = charmap.index(c1)
    y = charmap.index(c2)
    return (x * 95 + y) - 127

# Embedding
emb_data, emb_scales, emb_shapes = parse_laum(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_emb.laum')
print(f"emb scales: {emb_scales}")
print(f"emb shapes: {emb_shapes}")

# Token 86's row: int8 indices 85*16 to 85*16+15
emb_86 = np.array([dec(emb_data, 85*16+i) for i in range(16)]) * emb_scales[0]
print(f"\nDecoded emb[86] (1..8): {emb_86[:8]}")

# Now compute LN1 manually
mean = emb_86.mean()
var = emb_86.var()
print(f"mean: {mean}, var: {var}")
ln_out = (emb_86 - mean) / np.sqrt(var + 0.001)
print(f"ln_out (1..8): {ln_out[:8]}")

# Apply LN gain and bias from layer 0
L0_data, L0_scales, L0_shapes = parse_laum(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_l0.laum')
print(f"\nL0 shapes: {L0_shapes}")
print(f"L0 scales: {L0_scales}")

# Tensor offsets: shapes are pairs (d0, d1) so flat
# t0=ln1_g(16,1)=16, t1=ln1_b(16,1)=16, t2=wq(16,16)=256, t3=wk(16,16)=256, ...
# Tensor sizes = product of shape pairs
tensor_sizes = [L0_shapes[i*2] * L0_shapes[i*2+1] for i in range(len(L0_shapes)//2)]
offsets = np.cumsum([0] + tensor_sizes)
print(f"offsets: {offsets}")
print(f"tensor_sizes: {tensor_sizes}")

def get_tensor(idx):
    start = offsets[idx]
    end = offsets[idx+1]
    size = end - start
    return np.array([dec(L0_data, start+i) for i in range(size)]) * L0_scales[idx]

# ln1_g is tensor 0
ln1_g = get_tensor(0)
ln1_b = get_tensor(1)
print(f"ln1_g (1..8): {ln1_g[:8]}")
print(f"ln1_b (1..8): {ln1_b[:8]}")

# wq is tensor 2
wq = get_tensor(2).reshape(16, 16)
print(f"wq[1] (1..8): {wq[0][:8]}")

# Q proj
Q = ln_full @ wq
print(f"Q (1..8): {Q[:8]}")

# Attention with single token
score = (Q * K).sum() * 0.5  # 1/sqrt(4)
print(f"score: {score}")
attn_out = V  # softmax([score]) = [1]
wo = get_tensor(5).reshape(16, 16)
attn_proj = attn_out @ wo
print(f"attn_proj (1..8): {attn_proj[:8]}")

# Residual
x1 = emb_86 + attn_proj
print(f"x1 (after attn, 1..8): {x1[:8]}")

# LN2
ln2_g = get_tensor(6)
ln2_b = get_tensor(7)
mean2 = x1.mean()
var2 = x1.var()
ln2_out = (x1 - mean2) / np.sqrt(var2 + 0.001)
ln2_full = ln2_out * ln2_g + ln2_b
print(f"ln2_full (1..8): {ln2_full[:8]}")

# MLP
w1 = get_tensor(8).reshape(16, 16)
b1 = get_tensor(9)
w2 = get_tensor(10).reshape(16, 16)
b2 = get_tensor(11)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

h_pre = ln2_full @ w1 + b1
h_act = gelu(h_pre)
print(f"h_act (1..8): {h_act[:8]}")
mlp_out = h_act @ w2 + b2
print(f"mlp_out (1..8): {mlp_out[:8]}")

x_final = x1 + mlp_out
print(f"\nFINAL layer 0 output (1..8): {x_final[:8]}")