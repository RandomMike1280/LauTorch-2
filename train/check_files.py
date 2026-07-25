import os
files = [
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_020000.npz',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\model_20k.npz',
]
for f in files:
    if os.path.exists(f):
        print(f'{os.path.basename(f):25s} {os.path.getsize(f):>12,} bytes')
    else:
        print(f'{os.path.basename(f):25s} MISSING')
