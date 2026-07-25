"""Read exact bytes of chat.lau vocab line."""
with open('chat.lau', 'rb') as f:
    content = f.read()

# Find the line
for i, line in enumerate(content.split(b'\n')):
    if b'varol L=' in line:
        print(f"Line {i} ({len(line)} bytes):")
        for j, b in enumerate(line):
            c = chr(b)
            if 32 <= b <= 126:
                escaped = c
            else:
                escaped = f'\\x{b:02x}'
            print(f"  [{j:3d}] {b:3d} ({escaped:>4s})", end='')
            if j > 0 and (j-8) % 10 == 9:
                print()
        print()
        # Show the key part around position 10-15
        print("Key chars around position 8-16:")
        for j in range(8, min(16, len(line))):
            b = line[j]
            c = chr(b) if 32 <= b <= 126 else '?'
            print(f"  [{j}] byte={b} char={repr(c)}")
        break
