"""Check if weights.json has trained or random weights."""
import json, numpy as np

with open('www/weights.json') as f:
    data = json.load(f)

weights = data['weights']

print("Weight statistics (if std ~0.07-0.1 and mean ~0, weights are likely randomly initialized):\n")
for i, w in enumerate(weights):
    arr = np.array(w)
    flat = arr.flatten()
    print(f"W[{i+1}]: shape={arr.shape}, mean={flat.mean():.6f}, std={flat.std():.6f}, min={flat.min():.4f}, max={flat.max():.4f}")

# Check if weights look trained (non-trivial values) or random
print("\n\nChecking if embedding weights are meaningful:")
emb = np.array(weights[0])
print(f"Embedding norm per row (should be ~0.1-0.3 if trained):")
for row in range(0, 95, 10):
    norm = np.linalg.norm(emb[row])
    print(f"  char {row} ('{chr(row+32)}'): norm={norm:.4f}")
    
# Check if rows are distinct (trained) or mostly zeros
diffs = []
for r in range(1, 95):
    diff = np.abs(emb[r] - emb[r-1]).mean()
    diffs.append(diff)
print(f"\nMean abs diff between adjacent embedding rows: {np.mean(diffs):.4f}")
print(f"(Should be ~0.1-0.3 for trained, ~0 for random init)")
