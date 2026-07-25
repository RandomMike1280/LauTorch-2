"""Decode embedding from laum and compare to FP32"""
import numpy as np
import sys, re
sys.path.insert(0, 'train')
from quantize import _build_charmap

charmap = _build_charmap()

# Read raw laum file
with open(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\weights_emb.laum') as f:
    content = f.read()

# Find the d=... part
m = re.search(r'd="([^"]+)"', content)
data = m.group(1)
print("data len:", len(data))

# Find c=[...]
m = re.search(r'c=\{([^}]+)\}', content)
scale = float(m.group(1))
print("scale:", scale)

# Decode int8 (0-indexed)
def dec(i):
    p = i * 2
    c1 = data[p]
    c2 = data[p+1]
    x = charmap.index(c1)
    y = charmap.index(c2)
    return (x * 95 + y) - 127

# Token 86's row starts at int8 index 85*16 = 1360
w_fp32 = np.load(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights_fp32.npy', allow_pickle=True).item()
fp32_emb_86 = w_fp32['emb'][86]

print("\nDecoded int8 vs FP32 for token 86 (1..8):")
for i in range(16):
    v = dec(85*16 + i)
    print(f"  {i+1}: int8={v}, scaled={v*scale:.4f}, fp32={fp32_emb_86[i]:.4f}")