"""Check laum file for issues."""
with open('weights.laum') as f:
    content = f.read()

# Check for scientific notation
import re
sci = re.findall(r'-?\d+\.\d+e[+-]\d+', content)
print(f"Sci notation found: {len(sci)}")
if sci:
    print(f"Examples: {sci[:3]}")

# Check for -0.0 patterns (unwanted)
neg_zero = re.findall(r'-0\.0+[,\}]', content)
print(f"-0.0 patterns: {len(neg_zero)}")
if neg_zero:
    print(f"Examples: {neg_zero[:3]}")

# Check first 500 chars
print(f"\nFirst 500 chars:\n{content[:500]}")
