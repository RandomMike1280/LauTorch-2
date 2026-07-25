"""Export weights as a .laum module with literal float values."""
import json, os

with open('www/weights.json') as f:
    data = json.load(f)

print(f"Config: {data['config']}")
print(f"Weight arrays: {len(data['weights'])}")

# Vocab as list of chars
VOCAB = [chr(32 + i) for i in range(95)]
print(f"Vocab chars: {VOCAB[:5]}...")

# Format float
def fmt(v):
    if v == 0: return "0.0"
    s = f"{v:.20f}"
    s = s.rstrip('0').rstrip('.')
    if s.startswith('-0.0'): s = "0.0"
    return s

lines = []
lines.append("return {")
lines.append("  config={")
cfg = data['config']
lines.append(f"    vocab_size={cfg['vocab_size']},")
lines.append(f"    d_model={cfg['d_model']},")
lines.append(f"    n_layers={cfg['n_layers']},")
lines.append(f"    n_heads={cfg['n_heads']},")
lines.append(f"    head_dim={cfg['head_dim']},")
lines.append(f"    d_ff={cfg['d_ff']},")
lines.append(f"    ctx_len={cfg['ctx_len']},")
lines.append("  },")
# Vocab as list of single-char strings
lines.append("  vocab_chars={")
for c in VOCAB:
    if c == '"':
        lines.append('    "\\"",')
    elif c == '\\':
        lines.append('    "\\\\",')
    else:
        lines.append(f'    "{c}",')
lines.append("  },")
lines.append("  weights={")

for i, arr in enumerate(data['weights']):
    lines.append(f"    [{i+1}]={{")
    if isinstance(arr[0], list):
        for row in arr:
            row_str = ",".join(fmt(v) for v in row)
            lines.append(f"      {{{row_str}}},")
    else:
        row_str = ",".join(fmt(v) for v in arr)
        lines.append(f"      {{{row_str}}},")
    lines.append("    },")

lines.append("  },")
lines.append("}")

src = "\n".join(lines)

with open('weights.laum', 'w') as f:
    f.write(src)

print(f"Written weights.laum: {len(src):,} chars, {os.path.getsize('weights.laum'):,} bytes")
print(f"Lines: {len(lines)}")
