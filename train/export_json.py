"""Quantize a trained Lau Transformer to int8 and export to weights.json.

Format: a single JSON file loadable via http.jsonDecode in Lau.

Top-level keys:
- config: {d_model, n_layers, n_heads, head_dim, d_ff, ctx_len, vocab_size}
- names:  list of tensor names in order
- shapes: list of flat shape pairs [d0, d1, d0, d1, ...] (1D tensors have d1=1)
- scales: list of per-tensor float scales
- data:   flat list of int8 values (after dequantization: value = int8 * scale)
"""
import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN, VOCAB_SIZE,
)


def quantize_tensor(w):
    """Per-tensor symmetric int8 quantization."""
    qmax = 127.0
    abs_max = float(np.abs(w).max())
    if abs_max == 0:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = abs_max / qmax
    q = np.round(w / scale).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), 'weights', 'model_big.npz')
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), '..', 'weights.json')

    print(f"Loading {weights_path}...")
    with np.load(weights_path) as archive:
        params = {k: archive[k].copy() for k in archive.files}
    print(f"Loaded {len(params)} arrays")

    # Build the list of tensors in a stable order. Determine actual layer count
    # from the checkpoint, not from model.py (which may have been modified).
    layer_indices = sorted({
        int(k.split('.')[0][1:])
        for k in params if k.startswith('l') and k[1].isdigit()
    })
    n_layers_actual = max(layer_indices) + 1 if layer_indices else 0
    print(f"Model has {n_layers_actual} layers (from checkpoint)")

    # Build the list of tensors in a stable order
    names = []
    tensors = []
    names.append('emb')
    tensors.append(params['emb'])
    for i in range(n_layers_actual):
        for key in (f'l{i}.ln1_g', f'l{i}.ln1_b',
                    f'l{i}.wq', f'l{i}.wk', f'l{i}.wv', f'l{i}.wo',
                    f'l{i}.ln2_g', f'l{i}.ln2_b',
                    f'l{i}.w1', f'l{i}.b1', f'l{i}.w2', f'l{i}.b2'):
            if key not in params:
                raise KeyError(f"Checkpoint missing {key}")
            names.append(key)
            tensors.append(params[key])
    names.append('lnf_g')
    tensors.append(params['lnf_g'])
    names.append('lnf_b')
    tensors.append(params['lnf_b'])

    # Quantize each tensor
    int8_arrays = []
    scales = []
    shapes_flat = []
    total_params = 0
    for name, t in zip(names, tensors):
        q, s = quantize_tensor(t)
        int8_arrays.append(q)
        scales.append(s)
        # Use [] for 1D, [d0, d1] for 2D (Lau's jsonDecode returns lists for both)
        if q.ndim == 1:
            shapes_flat.append([int(q.shape[0])])
        else:
            shapes_flat.append([int(q.shape[0]), int(q.shape[1])])
        total_params += q.size

    flat_data = np.concatenate([q.flatten() for q in int8_arrays]).astype(np.int8)
    print(f"Total int8 values: {len(flat_data)} (params: {total_params})")

    # To save space, we serialize scales as a list of floats (Lau can read them as floats).
    # The int8 data is the bulk; we encode as a list of ints.
    out = {
        'config': {
            'd_model': D_MODEL,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'head_dim': HEAD_DIM,
            'd_ff': D_FF,
            'ctx_len': CTX_LEN,
            'vocab_size': VOCAB_SIZE,
        },
        'names': names,
        'shapes': shapes_flat,
        'scales': [float(s) for s in scales],
        'data': [int(x) for x in flat_data.tolist()],
    }

    print(f"Writing {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(out, f)
    size = os.path.getsize(out_path)
    print(f"Wrote {size:,} bytes")

    # Verify we can read it back
    print("\nVerifying readback...")
    with open(out_path, 'r') as f:
        rt = json.load(f)
    assert len(rt['data']) == len(flat_data)
    assert rt['config']['d_model'] == D_MODEL
    assert rt['config']['n_layers'] == N_LAYERS
    assert rt['config']['n_heads'] == N_HEADS
    print("Readback OK")

    # Print param count breakdown
    print("\nParameter breakdown:")
    for name, t in zip(names, tensors):
        print(f"  {name:20s} {str(t.shape):20s} {t.size:>10,}")
    print(f"  {'TOTAL':20s} {'':20s} {total_params:>10,}")


if __name__ == "__main__":
    main()
