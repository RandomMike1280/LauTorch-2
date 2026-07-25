"""Read chat.lau vocab bytes and extract exactly."""
with open('chat.lau', 'rb') as f:
    raw = f.read()

# Find the line
for i, line in enumerate(raw.split(b'\n')):
    if b'varol L=' in line:
        print(f"Line {i} (raw bytes): {line[:80]}")
        # Find first "
        start = line.find(b'"')
        # Find last " 
        end = line.rfind(b'"')
        print(f"  Start quote at byte {start}, end quote at byte {end}")
        vocab_bytes = line[start+1:end]
        print(f"  Vocab bytes ({len(vocab_bytes)}): {vocab_bytes[:20]}")
        print(f"  Last 5 bytes: {vocab_bytes[-5:]}")
        # Decode as utf-8
        try:
            vocab_str = vocab_bytes.decode('utf-8')
            print(f"  Decoded string length: {len(vocab_str)}")
            print(f"  First 10: {repr(vocab_str[:10])}")
            print(f"  Last 10: {repr(vocab_str[-10:])}")
            print(f"  All ASCII printable? {all(32 <= b <= 126 for b in vocab_bytes)}")
            if len(vocab_str) != 95:
                print(f"  ERROR: expected 95 chars, got {len(vocab_str)}!")
        except Exception as e:
            print(f"  Decode error: {e}")
        break
