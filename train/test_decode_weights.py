"""Decode all weights from laum files for comparison."""
import numpy as np
import sys, re
sys.path.insert(0, 'train')
from quantize import _build_charmap

charmap = _build_charmap()

def parse_laum(path):
    with open(path) as f:
        content = f.read()
    m_d = re.search(r'd="([^"]+)"', content)
    data = m_d.group(1)
    m_c = re.search(r'c=\{([^}]+)\}', content)
    scales = [float(s) for s in m_c.group(1).split(',')]
    m_s = re.search(r's=\{\s*([0-9,]+)\s*\}', content)
    shapes = [int(s) for s in m_s.group(1).split(',')]
    m_n = re.search(r'n=(\d+)', content)
    n = int(m_n.group(1))
    return data, scales, shapes, n

def dec(data, i):
    p = i * 2
    c1 = data[p]
    c2 = data[p+1]
    x = charmap.index(c1)
    y = charmap.index(c2)
    return (x * 95 + y) - 127

# Decode layer 0
data, scales, shapes, n = parse_laum(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_l0.laum')
print(f"L0 shapes: {shapes}, scales: {scales}")
# shapes shows the size of each tensor
# We have tensors: ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, w1, b1, w2, b2
# 12 tensors total
# Shapes alternate dim0, dim1
# t0=ln1_g (16), t1=ln1_b (16), t2=wq (16,16), t3=wk (16,16), t4=wv (16,16),
# t5=wo (16,16), t6=ln2_g (16), t7=ln2_b (16), t8=w1 (16,16), t9=b1 (16),
# t10=w2 (16,16), t11=b2 (16)

print(f"shapes count: {len(shapes)}")

# Cumulative offsets
offsets = [0]
total = 0
for s in shapes:
    offsets.append(total)
    total += s
print(f"Total weights: {total}, total chars: {total*2}, actual: {len(data)}")

# Each tensor's data
def get_tensor(idx):
    start = offsets[idx]
    end = offsets[idx+1]
    vals = [dec(data, start + i) for i in range(end - start)]
    return np.array(vals) * scales[idx]

# Tensor 2: wq (16x16)
wq = get_tensor(2).reshape(shapes[3], shapes[2])  # Note: shapes is flat [d0,d1,d0,d1,...]
print(f"\nWQ scale: {scales[2]}")
print("wq[0][:8]:", wq[0][:8])

# Layer 0 LN1
ln1_g = get_tensor(0)
ln1_b = get_tensor(1)
print(f"\nLN1_g (1..8): {ln1_g[:8]}")