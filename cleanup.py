import os
import glob

DIR = r'C:\Users\angel\OneDrive\Desktop\LauTorch-2'
patterns = ['test_*.lau', 'test.laum', 'test_*.py', 'check_*.py', 'count_*.py', 'audit_*.py', 'decode_*.py', 'final_check.py', 'check_*.py']

for pat in patterns:
    for f in glob.glob(os.path.join(DIR, pat)):
        os.remove(f)
        print(f"Removed: {os.path.basename(f)}")

print("\nRemaining .lau and .laum files:")
for f in os.listdir(DIR):
    if f.endswith('.lau') or f.endswith('.laum'):
        p = os.path.join(DIR, f)
        print(f"  {f}: {os.path.getsize(p)} bytes")