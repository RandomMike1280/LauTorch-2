"""Verify 64-char vocab construction."""
# ASCII 32-95 = 64 chars
VOCAB_64 = ''.join(chr(32 + i) for i in range(64))
print(f"Length: {len(VOCAB_64)}")
print(f"Chars: {repr(VOCAB_64)}")
print(f"Last: {repr(VOCAB_64[-1])} = ord {ord(VOCAB_64[-1])}")
# Check for missing chars
all_printable = ''.join(chr(i) for i in range(32, 127))
print(f"\nChars in 32-95: {len(all_printable)}")
print(f"Chars in 32-126: {len(''.join(chr(i) for i in range(32, 127)))}")
# Verify lowercase letters (97-122 -> IDs 65-90)
for c in 'hello':
    idx = ord(c) - 32
    in_vocab = idx < 64
    print(f"{repr(c)}: ASCII {ord(c)} -> ID {idx}, in vocab: {in_vocab}")
