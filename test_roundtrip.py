"""Test round-trip."""
import sys
sys.path.insert(0, 'train')
from model import VOCAB_SIZE, VOCAB
from train import encode

test = "Hello world! What's 1+1? Paris"
ids = encode(test)
back = ''.join(VOCAB[i] for i in ids if 0 <= i < len(VOCAB))
print(f'Original:  {repr(test)}')
print(f'Decoded:   {repr(back)}')
print(f'Match: {test == back}')

# Test all chars
mismatch = []
for c in map(chr, range(32, 127)):
    ids = encode(c)
    back = ''.join(VOCAB[i] for i in ids if 0 <= i < len(VOCAB))
    if back != c:
        mismatch.append((c, back))
print(f'Mismatches: {len(mismatch)}')
if mismatch:
    print(f'First few: {mismatch[:5]}')
print(f'Vocab size: {VOCAB_SIZE}')
print(f'Vocab length: {len(VOCAB)}')
