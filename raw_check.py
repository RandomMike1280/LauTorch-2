"""Check raw bytes of chat.lau line 20."""
with open('chat.lau', 'rb') as f:
    lines = f.read().split(b'\n')

# Line 20 (0-indexed: line 19)
line = lines[19]
print(f"Line 20 raw: {line}")
print(f"Length: {len(line)}")

# Find first and last "
first = line.find(b'"')
last = line.rfind(b'"')
print(f"First quote at {first}, last quote at {last}")
vocab_bytes = line[first+1:last]
print(f"Vocab bytes: {len(vocab_bytes)}")
print(f"First 5: {vocab_bytes[:5]}")
print(f"Last 5: {vocab_bytes[-5:]}")

# Decode as ascii
try:
    vocab = vocab_bytes.decode('ascii')
    print(f"Decoded length: {len(vocab)}")
    print(f"Expected: 95")
    if len(vocab) != 95:
        print(f"MISMATCH!")
    # Show char at each position
    for i, c in enumerate(vocab[:10]):
        print(f"  [{i}] {repr(c)} (ASCII {ord(c)})")
except Exception as e:
    print(f"Decode error: {e}")

# Python string comparison
L_py = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
print(f"\nPython string length: {len(L_py)}")
if len(vocab) == len(L_py):
    print("Lengths match!")
    for i, (a, b) in enumerate(zip(vocab, L_py)):
        if a != b:
            print(f"MISMATCH at {i}: file={repr(a)} (ASCII {ord(a)}), python={repr(b)} (ASCII {ord(b)})")
else:
    print(f"Lengths differ: file={len(vocab)}, python={len(L_py)}")
