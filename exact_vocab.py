"""Exactly extract the vocab from chat.lau as Python reads it."""
import re

with open('chat.lau', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find the line
for line in content.split('\n'):
    if line.startswith('varol L='):
        # Extract the value after '='
        rest = line[8:]  # after 'varol L='
        # Find first and last double-quote
        first = rest.find('"')
        last = rest.rfind('"')
        vocab = rest[first+1:last]
        print(f"Vocab string (as Python reads it):")
        print(f"  Length: {len(vocab)}")
        print(f"  First 10: {repr(vocab[:10])}")
        print(f"  Last 10: {repr(vocab[-10:])}")
        print(f"  Char 2: {repr(vocab[2])} (ASCII {ord(vocab[2])})")
        print(f"  Char 3: {repr(vocab[3])} (ASCII {ord(vocab[3])})")
        
        # Compare to what we expect
        expected = ''.join(chr(32+i) for i in range(95))
        print(f"\nExpected length: {len(expected)}")
        print(f"Match: {vocab == expected}")
        if vocab != expected:
            # Find first mismatch
            for i in range(min(len(vocab), len(expected))):
                if vocab[i] != expected[i]:
                    print(f"  First mismatch at pos {i}: file={repr(vocab[i])} (ASCII {ord(vocab[i])}), expected={repr(expected[i])} (ASCII {ord(expected[i])})")
                    break
            if len(vocab) != len(expected):
                print(f"  Length diff: file={len(vocab)}, expected={len(expected)}")
        break
