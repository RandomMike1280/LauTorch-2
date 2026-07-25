import os
import time
path = r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\model.py'
mtime = os.path.getmtime(path)
print(f'model.py mtime: {time.ctime(mtime)} (now: {time.ctime()})')
print(f'age: {time.time() - mtime:.0f}s')

path = r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\checkpoint_020000.npz'
mtime = os.path.getmtime(path)
print(f'checkpoint_020000.npz mtime: {time.ctime(mtime)}')
print(f'age: {time.time() - mtime:.0f}s')
