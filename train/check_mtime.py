import os
import time
path = r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\progress.txt'
print(f'progress.txt mtime: {time.time() - os.path.getmtime(path):.0f}s ago')
path = r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_020000.npz'
print(f'checkpoint_020000.npz mtime: {time.time() - os.path.getmtime(path):.0f}s ago')
