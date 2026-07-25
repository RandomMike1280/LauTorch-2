import os
import time
files = [
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\model.py',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\train_big.py',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\export_json.py',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\verify_json.py',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_010000.npz',
    r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_020000.npz',
]
for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        print(f'{os.path.basename(f):25s} {time.ctime(mtime)}')
    else:
        print(f'{os.path.basename(f):25s} MISSING')
