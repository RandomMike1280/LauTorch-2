"""Test float formatting."""
import json

def fmt(v):
    if v == 0: return "0.0"
    s = f"{v:.20f}"
    s = s.rstrip('0').rstrip('.')
    if s.startswith('-0.0'): s = "0.0"
    return s

# Test
vals = [0.0, 1.0, -1.0, 1e-10, -1e-10, 1.234567890123456789]
for v in vals:
    print(f"{v} -> {fmt(v)}")

# Check the actual JSON values
with open('www/weights.json') as f:
    data = json.load(f)
small_vals = [v for arr in data['weights'] for row in (arr if isinstance(arr[0], list) else [arr]) for v in row if abs(v) < 1e-8 and v != 0]
print(f"\nSmall non-zero values: {len(small_vals)}")
if small_vals:
    print(f"Examples: {small_vals[:5]}")
    for v in small_vals[:5]:
        print(f"  {v} -> {fmt(v)}")
