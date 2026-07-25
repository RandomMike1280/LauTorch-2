"""Quantize the trained model to int8 and split into 3 .laum files.

Each .laum file is a Lau script that returns a list of base-100 encoded weight
strings plus the per-tensor scales and shapes.

Encoding:
- int8 values in range [-127, 127] -> offset to [0, 254] -> stored as 2 chars
  - high byte: value // 100 (0-2)
  - low byte: value % 100 (0-99)
  - special encoding for negative: map [-127, 0] to [0, 127] and [1, 127] to [128, 254]

Each .laum file returns:
{
  shapes: [{V, D}, ...],  # shapes for each tensor
  scales: [s1, s2, ...],  # scales for each tensor
  data: [string1, ...],   # base-100 encoded weights
  meta: {n_tensors, total_params}
}
"""
import os
import json
import numpy as np


def quantize_tensor(w, target='int8'):
    """Per-tensor symmetric int8 quantization."""
    qmax = 127.0
    abs_max = np.abs(w).max()
    if abs_max == 0:
        scale = 1.0
    else:
        scale = abs_max / qmax
    q = np.round(w / scale).astype(np.int32)
    q = np.clip(q, -127, 127)
    return q, float(scale)


def _build_charmap():
    """Build a unique 100-char printable mapping.

    Use ASCII 33-126 (94 chars) for 0-93, plus 6 extras from 32 (' ').
    But space isn't great. Let's use 33-126 (94) + chr(1) through chr(6) won't work.
    Use 33-126 (94) + a couple of punctuation above 126 won't work either.
    Simplest: use chr(33) to chr(99) and chr(101) to chr(133).
    Actually let's just use a clean mapping:
      0-9: '0'-'9' (10 chars)
      10-35: 'a'-'z' (26 chars)
      36-61: 'A'-'Z' (26 chars)
      62-87: '!' to '8' (offset 33, but skip dupes)
      ...
    Cleanest: use chr(33) through chr(132)... but that's 100 chars exactly.
    chr(33)='!' to chr(132)=non-ascii. Let me just use 100 sequential ASCII codes.
    """
    # Use chr(33) through chr(132). chr(33)='!' to chr(126)='~' (94 chars),
    # then chr(127) through chr(132) are extended ASCII but still valid in unicode.
    # Actually, safe ASCII printable is 32-126 = 95 chars. We need 100.
    # Use chr(32) through chr(131) — but chr(127) is DEL, chr(128-159) are control.
    # Best: use 33-126 (94 chars) for 0-93, then 4 special chars.
    # Let's just use 27+27+10+10+10+10+6 = 100:
    # 0-9: '0'-'9'
    # 10-35: 'a'-'z' (26)
    # 36-61: 'A'-'Z' (26)
    # 62-71: '!' to '(' (9 chars: ! " # $ % & ' ( )
    # 72-81: '*' to '1' (10 chars: * + , - . / 0 1 2 3 — wait 0-9 are taken)
    # This gets messy. Let's use a totally clean ASCII table.
    #
    # Final clean mapping using chr(33)..chr(132):
    # 0-93: chr(33) ('!') to chr(126) ('~') in order
    # 94-99: chr(127) to chr(132) — these are non-printable but Lau should handle them
    # but we want printable. Let me use Unicode characters.
    # Actually, the safest is to use 95 printable ASCII chars (33-126) for 0-94,
    # then we have only 5 more. Total 100.
    #
    # Try yet another approach: use chr(33) to chr(132) which includes DEL and
    # the first 6 control chars. These are valid Unicode characters but may cause
    # issues in Lau source. Let me test.
    #
    # Simplest reliable: use a base-95 encoding (one char per value), and use
    # chr(33) ('!') to chr(126) ('~') with chars wrapping around.
    # Then we need 2 chars per int8 value, mapping value to (high, low) where
    # 0 <= high < 95 and 0 <= low < 95.
    # Use chr(32) (' ' = id 0) through chr(126) ('~' = id 94)
    # This matches Lau's alphabet string which has space at position 1
    chars = []
    for i in range(95):
        chars.append(chr(32 + i))  # ' ' to '~'
    return chars


CHARS = _build_charmap()
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}


def encode_int8(q):
    """Encode int8 [-127, 127] to base-95 string (2 chars per value).

    Map [-127, 127] -> [0, 254]:
      -127 -> 0, -126 -> 1, ..., 0 -> 127, ..., 127 -> 254
    Then encode 0-254 as (high, low) where high = v // 95, low = v % 95.
    """
    v = q + 127  # map to [0, 254]
    high = v // 95  # 0, 1, or 2
    low = v % 95    # 0-94
    return ''.join(CHARS[h] + CHARS[l] for h, l in zip(high, low))


def decode_int8(s):
    """Reverse of encode_int8."""
    out = []
    for i in range(0, len(s), 2):
        h = ord(s[i]) - 32
        l = ord(s[i+1]) - 32
        if h < 0 or h >= 95: h = 0
        if l < 0 or l >= 95: l = 0
        v = h * 95 + l - 127
        out.append(v)
    return np.array(out, dtype=np.int32)


def main():
    weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'model.npz')
    out_dir = os.path.join(os.path.dirname(__file__), '..')
    w = np.load(weights_path)
    params = {k: w[k].copy() for k in w.files}
    print(f"Loaded {len(params)} arrays")

    # We split into 3 .laum files. Plan:
    # weights_emb.laum: embedding (96x16=1536) -> ~3072 bytes encoded
    # weights_l0.laum: layer 0 weights (~2176 params) -> ~4352 bytes
    # weights_l1.laum: layer 1 weights + final LN (~2208 params) -> ~4416 bytes
    # Each file also needs metadata: shape pairs, scale values

    # Actually, let me compute exact sizes
    def estimate(tensors):
        total_params = sum(t.size for t in tensors)
        encoded_bytes = total_params * 2  # 2 chars per int8
        return encoded_bytes

    # Group 1: embeddings
    emb_tensors = [params['emb']]
    emb_param_count = sum(t.size for t in emb_tensors)
    print(f"emb: {emb_param_count} params, est {estimate(emb_tensors)} bytes")

    # Group 2: layer 0
    l0_tensors = [
        params['l0.ln1_g'], params['l0.ln1_b'],
        params['l0.wq'], params['l0.wk'], params['l0.wv'], params['l0.wo'],
        params['l0.ln2_g'], params['l0.ln2_b'],
        params['l0.w1'], params['l0.b1'], params['l0.w2'], params['l0.b2'],
    ]
    l0_param_count = sum(t.size for t in l0_tensors)
    print(f"l0: {l0_param_count} params, est {estimate(l0_tensors)} bytes")

    # Group 3: layer 1 + final LN
    l1_tensors = [
        params['l1.ln1_g'], params['l1.ln1_b'],
        params['l1.wq'], params['l1.wk'], params['l1.wv'], params['l1.wo'],
        params['l1.ln2_g'], params['l1.ln2_b'],
        params['l1.w1'], params['l1.b1'], params['l1.w2'], params['l1.b2'],
        params['lnf_g'], params['lnf_b'],
    ]
    l1_param_count = sum(t.size for t in l1_tensors)
    print(f"l1+lnf: {l1_param_count} params, est {estimate(l1_tensors)} bytes")

    # Total
    total = emb_param_count + l0_param_count + l1_param_count
    print(f"Total: {total} params")

    def build_module(tensors, names):
        """Build a Lau module string that returns a list of [shapes, scales, data, names]."""
        shapes = []
        scales = []
        encoded = []
        for t, name in zip(tensors, names):
            q, s = quantize_tensor(t)
            s_flat = q.flatten()
            shapes.append([int(t.shape[0]), int(t.shape[1]) if len(t.shape) > 1 else 1])
            scales.append(s)
            encoded.append(encode_int8(s_flat))
        # Sanity check
        for i, (t, name) in enumerate(zip(tensors, names)):
            assert len(encoded[i]) == t.size * 2, f"{name}: expected {t.size*2} got {len(encoded[i])}"
        # Compact Lau format:
        # shapes as a flat list: {V, D, V, D, ...} (defaults to 1D if D=1)
        # scales as a flat list of floats
        # data as a single concatenated string in order
        # We use ONE table instead of three so the Laum file is as small as possible
        flat_shapes = []
        for shape in shapes:
            flat_shapes.extend(shape)
        # Use compact repr for scales: 5 decimal places
        scale_strs = [f'{s:.6f}' for s in scales]
        # Concatenate data into one string
        all_data = ''.join(encoded)
        # Build the source
        # Named keys required for Lau table syntax
        lines = [
            'return {',
            '  s={' + ','.join(str(x) for x in flat_shapes) + '},',
            '  c={' + ','.join(scale_strs) + '},',
            '  d="' + all_data.replace('\\', '\\\\').replace('"', '\\"') + '",',
            '  n=' + str(len(tensors)),
            '}',
        ]
        return '\n'.join(lines)

    # Write the modules
    emb_names = ['emb']
    l0_names = ['l0.ln1_g', 'l0.ln1_b', 'l0.wq', 'l0.wk', 'l0.wv', 'l0.wo',
                'l0.ln2_g', 'l0.ln2_b', 'l0.w1', 'l0.b1', 'l0.w2', 'l0.b2']
    l1_names = ['l1.ln1_g', 'l1.ln1_b', 'l1.wq', 'l1.wk', 'l1.wv', 'l1.wo',
                'l1.ln2_g', 'l1.ln2_b', 'l1.w1', 'l1.b1', 'l1.w2', 'l1.b2',
                'lnf_g', 'lnf_b']

    emb_source = build_module(emb_tensors, emb_names)
    l0_source = build_module(l0_tensors, l0_names)
    l1_source = build_module(l1_tensors, l1_names)

    print(f"\nemb.laum size: {len(emb_source)} bytes")
    print(f"l0.laum size: {len(l0_source)} bytes")
    print(f"l1.laum size: {len(l1_source)} bytes")

    # Write to disk
    out_files = {
        'weights_emb.laum': emb_source,
        'weights_l0.laum': l0_source,
        'weights_l1.laum': l1_source,
    }
    for fname, src in out_files.items():
        path = os.path.join(out_dir, fname)
        with open(path, 'w', encoding='ascii') as f:
            f.write(src)
        print(f"Wrote {path}")

    # Sanity check: parse back and verify
    print("\nVerifying decode...")
    for tensor, name in zip(emb_tensors, emb_names):
        q, s = quantize_tensor(tensor)
        encoded = encode_int8(q.flatten())
        decoded = decode_int8(encoded)
        diff = np.abs(decoded - q.flatten()).max()
        print(f"  {name}: max diff = {diff}")


if __name__ == "__main__":
    main()
