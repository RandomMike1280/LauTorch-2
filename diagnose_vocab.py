"""Diagnose vocab issue."""
# Check chat.lau vocab
L = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
print(f'chat.lau vocab size: {len(L)}')
print(f'First: {repr(L[0])}, Last: {repr(L[-1])} = ord {ord(L[-1])}')
print(f'Char at index 95: {repr(L[95]) if len(L) > 95 else "missing"}')
print(f'Char at index 63: {repr(L[63])}')

# Show corruption
print('\nCorruption map (ID -> decoded):')
for c in 'Hello World':
    idx = ord(c) - 32
    wrapped = idx % len(L)
    decoded = L[wrapped]
    ok = (decoded == c)
    if not ok:
        print(f'  {repr(c)} -> ID {idx} -> wrapped {wrapped} -> decoded {repr(decoded)} CORRUPT')
