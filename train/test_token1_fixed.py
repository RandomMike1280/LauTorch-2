import re, sys
import numpy as np
sys.path.insert(0, 'train')

with open(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_emb.laum') as f:
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
emb_scale = float(re.search(r'c=\{([^}]+)\}', c).group(1))

with open(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_l0.laum') as f:
    c = f.read()
i = c.find('d="') + 3
L0data = []
while i < len(c):
    ch = c[i]
    if ch == '\\':
        i += 1
        L0data.append(c[i])
    elif ch == '"':
        break
    else:
        L0data.append(ch)
    i += 1
L0data = ''.join(L0data)
L0_scales = [float(s) for s in re.search(r'c=\{([^}]+)\}', c).group(1).split(',')]
L0_shapes = [int(s) for s in re.search(r's=\{\s*([0-9,]+)\s*\}', c).group(1).split(',')]

from quantize import _build_charmap
charmap = _build_charmap()

def dec(d, i):
    p = i * 2
    x = charmap.index(d[p])
    y = charmap.index(d[p+1])
    return (x * 95 + y) - 127

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

emb_86 = np.array([dec(data, 85*16+j) for j in range(16)]) * emb_scale

tensor_sizes = [L0_shapes[i*2] * L0_shapes[i*2+1] for i in range(len(L0_shapes)//2)]
offsets = [0]
for s in tensor_sizes:
    offsets.append(offsets[-1] + s)

def get_t(idx):
    start = offsets[idx]
    size = tensor_sizes[idx]
    return np.array([dec(L0data, start+j) for j in range(size)]) * L0_scales[idx]

# Tensors:
# 0=ln1_g, 1=ln1_b, 2=wq, 3=wk, 4=wv, 5=wo, 6=ln2_g, 7=ln2_b, 8=w1, 9=b1, 10=w2, 11=b2
ln1_g = get_t(0)
ln1_b = get_t(1)
wq = get_t(2).reshape(16, 16)
wk = get_t(3).reshape(16, 16)
wv = get_t(4).reshape(16, 16)
wo = get_t(5).reshape(16, 16)
ln2_g = get_t(6)
ln2_b = get_t(7)
w1 = get_t(8).reshape(16, 16)
b1 = get_t(9)
w2 = get_t(10).reshape(16, 16)
b2 = get_t(11)

print(f"b1 (1..4): {b1[:4]}")
print(f"b2 (1..4): {b2[:4]}")

# LN1
mean = emb_86.mean()
var = emb_86.var()
ln_out = (emb_86 - mean) / np.sqrt(var + 0.001)
ln_full = ln_out * ln1_g + ln1_b

# Q, K, V with FIXED indexing (x @ W)
Q = ln_full @ wq
K = ln_full @ wk
V = ln_full @ wv

print(f"Q (1..4): {Q[:4]}")
print(f"K (1..4): {K[:4]}")
print(f"V (1..4): {V[:4]}")

# Attention (single token)
score = (Q * K).sum() * 0.5
attn_out = V  # softmax([score]) = [1]

# Output projection with FIXED indexing (attn @ wo)
attn_proj = attn_out @ wo
print(f"attn_proj (1..8): {attn_proj[:8]}")

# Residual
x1 = emb_86 + attn_proj

# LN2
mean2 = x1.mean()
var2 = x1.var()
ln2_out = (x1 - mean2) / np.sqrt(var2 + 0.001)
ln2_full = ln2_out * ln2_g + ln2_b

# MLP with FIXED indexing
h_pre = ln2_full @ w1 + b1
h_act = gelu(h_pre)
mlp_out = h_act @ w2 + b2

x_final = x1 + mlp_out
print(f"\nFINAL layer 0 output (1..8): {x_final[:8]}")