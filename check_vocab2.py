"""Check vocab length."""
# The exact string from chat.lau
L = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
print(f'chat.lau vocab length: {len(L)}')
print(f'First 5: {repr(L[:5])}')
print(f'Last 5: {repr(L[-5:])}')
# Check what chars are present
missing = []
for i in range(32, 127):
    c = chr(i)
    if c not in L:
        missing.append((i, c))
if missing:
    print(f'Missing {len(missing)} chars: {missing[:10]}')
else:
    print('All chars 32-126 present!')
# Check for duplicate chars
seen = set()
dups = []
for c in L:
    if c in seen:
        dups.append(c)
    seen.add(c)
if dups:
    print(f'Duplicates: {dups}')
else:
    print('No duplicates')
