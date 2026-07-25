"""Verify chat.lau vocab."""
import re

with open('chat.lau', 'r') as f:
    src = f.read()

# Find the vocab line
for line in src.splitlines():
    if line.strip().startswith('varol L='):
        print(f"Full line: {repr(line)}")
        print(f"Line length: {len(line)}")
        # Try to extract the string value
        # Lua double-quoted string: find first " and last "
        start = line.find('"')
        end = line.rfind('"')
        if start >= 0 and end > start:
            vocab = line[start+1:end]
            print(f"\nExtracted vocab:")
            print(f"  Length: {len(vocab)}")
            print(f"  First 10: {repr(vocab[:10])}")
            print(f"  Last 10: {repr(vocab[-10:])}")
            print(f"  Char at pos 0: {repr(vocab[0])}, ASCII: {ord(vocab[0])}")
            print(f"  Char at pos 94: {repr(vocab[94])}, ASCII: {ord(vocab[94])}")
            print(f"  Char at pos 95: {repr(vocab[95])}, ASCII: {ord(vocab[95])}")
            print(f"  Char at pos 96: {repr(vocab[96])}, ASCII: {ord(vocab[96])}")
        else:
            print(f"Could not extract: start={start}, end={end}")
            print(f"Line after varol L=: {repr(line[8:])}")
        break

# Also check if model expects 95 or 95 chars
print("\nExpected:")
print(f"  ASCII 32-126 = {126-32+1} chars")
print(f"  ASCII 32-95 = {95-32+1} chars")
