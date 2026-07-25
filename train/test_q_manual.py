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

# Token 86
emb_86 = np.array([dec(data, 85*16+j) for j in range(16)]) * emb_scale

# LN1
mean = emb_86.mean()
var = emb_86.var()
ln_out = (emb_86 - mean) / np.sqrt(var + 0.001)

tensor_sizes = [L0_shapes[i*2] * L0_shapes[i*2+1] for i in range(len(L0_shapes)//2)]
offsets = [0]
for s in tensor_sizes:
    offsets.append(offsets[-1] + s)

def get_t(idx):
    start = offsets[idx]
    size = tensor_sizes[idx]
    return np.array([dec(L0data, start+j) for j in range(size)]) * L0_scales[idx]

ln1_g = get_t(0)
ln1_b = get_t(1)
ln_full = ln_out * ln1_g + ln1_b

wq = get_t(2).reshape(16, 16)
Q = ln_full @ wq
print(f"Python Q (1..8): {Q[:8]}")

# Manually compute Q[1]:
# Q[0] = sum over j of ln_full[j] * wq[0][j]
q0 = sum(ln_full[j] * wq[0][j] for j in range(16))
print(f"Python Q[1] manually: {q0}")

# Print each term
for j in range(8):
    print(f"  j={j+1}: ln_full={ln_full[j]:.4f}, wq[1][{j+1}]={wq[0][j]:.4f}, product={ln_full[j]*wq[0][j]:.4f}")