import numpy as np
with np.load(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_010000.npz') as w:
    print('10k keys:', sorted(w.files))
    print(f'10k total params: {sum(w[k].size for k in w.files):,}')
    for k in sorted(w.files):
        print(f'  {k:20s} {str(w[k].shape):20s} {w[k].size:>10,}')

print()
with np.load(r'C:\Users\angel\OneDrive\Desktop\LauTorch-2\train\weights\checkpoint_020000.npz') as w:
    print('20k keys:', sorted(w.files))
    print(f'20k total params: {sum(w[k].size for k in w.files):,}')
    for k in sorted(w.files):
        print(f'  {k:20s} {str(w[k].shape):20s} {w[k].size:>10,}')
