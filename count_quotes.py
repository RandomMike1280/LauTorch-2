"""Count quotes in chat.lau vocab line."""
with open('chat.lau', 'r') as f:
    src = f.read()

for line in src.splitlines():
    if 'varol L=' in line:
        print(f"Line: {repr(line)}")
        # Count double-quotes
        qcount = line.count('"')
        print(f"Double-quote count: {qcount}")
        # Show positions of quotes
        for i, c in enumerate(line):
            if c == '"':
                print(f"  Quote at pos {i}: context = {repr(line[max(0,i-3):i+4])}")
        # Extract: everything between FIRST and LAST quote
        first = line.find('"')
        last = line.rfind('"')
        vocab_raw = line[first+1:last]
        print(f"Between quotes: len={len(vocab_raw)}, repr={repr(vocab_raw[:20])}")
        break
