"""Check Python string length of the chat.lau vocab."""
# The exact string from chat.lau line 20
L = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
print(f"String length: {len(L)}")
print(f"First 5: {repr(L[:5])}")
print(f"Last 5: {repr(L[-5:])}")
# Check each char
for i, c in enumerate(L):
    if ord(c) != 32 + i:
        print(f"Mismatch at position {i}: expected ASCII {32+i}, got {ord(c)} ({repr(c)})")
# What are positions 0-4?
for i in range(5):
    print(f"  [{i}] {repr(L[i])} (ASCII {ord(L[i])})")
