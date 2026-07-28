"""Train the tiny transformer on the Q&A dataset.

Pure NumPy. AdamW with linear warmup + cosine decay. We train for many steps
on a stream of short windows from the corpus.
"""
import os
import sys
import time
import json
import numpy as np

# Make model importable
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    init_weights, forward_full, backward_full, param_size, VOCAB,
    VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, D_FF, CTX_LEN
)


def encode(text, vocab_size=VOCAB_SIZE):
    """Map ASCII chars to token ids 0..vocab_size-1.

    Use a 95-char vocab: all printable ASCII (32-126).
    ' ' (32) -> 0, '!' (33) -> 1, ..., '~' (126) -> 94.
    """
    out = np.frombuffer(text.encode('ascii'), dtype=np.uint8).astype(np.int64)
    out = out - 32  # shift so ' ' (32) -> 0, '~' (126) -> 94
    out = np.clip(out, 0, vocab_size - 1)
    return out


def decode(ids, vocab_size=VOCAB_SIZE):
    chars = []
    for i in ids:
        idx = int(i)
        if 0 <= idx < len(VOCAB):
            chars.append(VOCAB[idx])
    return ''.join(chars)


def get_batch(data, batch_size, ctx_len):
    """Sample a batch of windows from data."""
    max_start = len(data) - ctx_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s:s + ctx_len] for s in starts])
    y = np.stack([data[s + 1:s + ctx_len + 1] for s in starts])
    return x.astype(np.int32), y.astype(np.int32)


def adamw_step(params, grads, m, v, step, lr, beta1=0.9, beta2=0.999, eps=1e-8,
                weight_decay=0.01, decoupled_wd=True, clip_val=None, max_norm=None):
    """AdamW with optional weight regularization.

    Args:
        weight_decay: L2 / decoupled weight decay coefficient.
        decoupled_wd: If True (default), apply weight decay as a separate step
            (Loshchilov & Hutter 2019). If False, mix it into the gradient
            (legacy "L2 regularization" form).
        clip_val: If set, hard-clip every weight value to [-clip_val, +clip_val]
            after the update. Cheap and guarantees a bound, but discontinuous.
        max_norm: If set, scale each 2D weight matrix so its Frobenius norm is
            at most max_norm after the update. Smooth, prevents norm explosion.
    """
    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
        m_hat = m[k] / (1 - beta1 ** step)
        v_hat = v[k] / (1 - beta2 ** step)
        if decoupled_wd:
            params[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps))
            params[k] -= lr * weight_decay * params[k]
        else:
            params[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * params[k])
        if clip_val is not None:
            params[k] = np.clip(params[k], -clip_val, clip_val)
        if max_norm is not None and params[k].ndim >= 2:
            pnorm = np.linalg.norm(params[k])
            if pnorm > max_norm:
                params[k] *= max_norm / pnorm


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train(text_path, steps=3000, batch_size=8, ctx_len=CTX_LEN, max_lr=3e-3, min_lr=3e-4, warmup=100,
          seed=42, log_every=100, save_path=None, clip_val=9.0, max_norm=None):
    np.random.seed(seed)
    print(f"Loading {text_path}...")
    with open(text_path, 'r', encoding='ascii') as f:
        text = f.read()
    print(f"Loaded {len(text)} chars")
    data = encode(text)
    print(f"Encoded {len(data)} tokens, vocab range: {data.min()}..{data.max()}")

    params = init_weights(seed=seed)
    print(f"Model has {param_size(params)} params")

    m = {k: np.zeros_like(p) for k, p in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}

    # Test prompts for sanity check
    test_prompts = ["Human: Hello\nBot:", "Human: What's 1+1\nBot:", "Human: What's the capital of France?\nBot:"]

    log = {'loss': [], 'step': [], 'lr': [], 'samples': []}
    start = time.time()
    for step in range(steps):
        x, y = get_batch(data, batch_size, ctx_len)
        logits, cache = forward_full(params, x)
        # Compute loss
        log_probs = logits - logits.max(axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
        nll = -log_probs[np.arange(batch_size)[:, None], np.arange(ctx_len)[None, :], y]
        loss = nll.mean()
        if step % log_every == 0:
            lr = get_lr(step, warmup, steps, max_lr, min_lr)
            elapsed = time.time() - start
            print(f"step {step:4d} loss={loss:.4f} lr={lr:.5f} elapsed={elapsed:.1f}s")
            log['loss'].append(float(loss))
            log['step'].append(step)
            log['lr'].append(float(lr))
            # Sample a few test prompts
            if step % (log_every * 5) == 0:
                samples = []
                for p in test_prompts:
                    s = generate(params, p, max_new=30, temperature=0.5)
                    samples.append(s)
                log['samples'].append({'step': step, 'samples': samples})
                for s in samples:
                    print(f"  > {repr(s)}")
        grads = backward_full(params, logits, y, cache)
        lr = get_lr(step, warmup, steps, max_lr, min_lr)
        adamw_step(params, grads, m, v, step + 1, lr, clip_val=clip_val, max_norm=max_norm)

    print(f"Training complete. Final loss: {loss:.4f}")
    log['final_loss'] = float(loss)

    # Final test
    print("\nFinal samples:")
    for p in test_prompts:
        s = generate(params, p, max_new=40, temperature=0.5)
        print(f"  > {repr(s)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **{k: v for k, v in params.items()})
        with open(save_path + '.log.json', 'w') as f:
            json.dump(log, f, indent=2)
        print(f"Saved weights to {save_path}")
    return params, log


def generate(params, prompt, max_new=50, temperature=1.0, ctx_len=CTX_LEN):
    """Generate text from a prompt using KV-cache-less simple autoregressive."""
    tokens = encode(prompt).tolist()
    for _ in range(max_new):
        # Use last ctx_len tokens
        ctx = tokens[-ctx_len:]
        x = np.array([ctx], dtype=np.int32)
        logits, _ = forward_full(params, x)
        # Get last position logits
        last_logits = logits[0, -1, :]
        if temperature <= 0:
            next_id = int(last_logits.argmax())
        else:
            scaled = last_logits / temperature
            scaled = scaled - scaled.max()
            probs = np.exp(scaled)
            probs = probs / probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        tokens.append(next_id)
        # Stop on newline (decoded as char 10 -> encoded as 10-32 = -22, out of vocab)
        # Actually we mapped ' ' to 0, '\n' maps to 10-32 = -22 which we clip to 0 (' ')
        # So we need a different stop signal. Let's stop on newline byte 10 which we encode as 0 (space).
        # Better: stop on '\n' in the decoded text.
        if next_id + 32 == 10:  # newline
            break
        if len(tokens) > 200:
            break
    return decode(tokens)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(__file__), 'chat.txt'))
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), 'weights', 'model.npz'))
    parser.add_argument('--clip-val', type=float, default=9.0, help='Hard clip weights to [-clip, +clip]; set <=0 to disable.')
    parser.add_argument('--max-norm', type=float, default=None, help='If set, rescale each 2D weight matrix to this Frobenius norm.')
    args = parser.parse_args()
    train(args.data, steps=args.steps, batch_size=args.batch, max_lr=args.lr, seed=args.seed,
          save_path=args.out, clip_val=(args.clip_val if args.clip_val > 0 else None),
          max_norm=args.max_norm)
