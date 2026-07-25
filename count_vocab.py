"""Count chars in chat.lau vocab."""
with open('chat.lau', 'r') as f:
    src = f.read()
# Find the vocab line
for line in src.splitlines():
    if line.startswith('varol L='):
        # Extract the string
        rest = line[8:]  # after 'varol L='
        # Find matching quotes
        if rest.startswith('"'):
            end = rest.rfind('"')
            vocab = rest[1:end]
        elif rest.startswith("'"):
            end = rest.rfind("'")
            vocab = rest[1:end]
        print(f"Found vocab: length={len(vocab)}")
        print(f"First 10: {repr(vocab[:10])}")
        print(f"Last 10: {repr(vocab[-10:])}")
        # Count each character
        for i, c in enumerate(vocab):
            if i >= 100:
                break
            pass
        break
