"""verify_chunks.py — sanity-check the chunk round-trip.

Reassembles the chunks in order and confirms:
  - total byte length matches the manifest's total_bytes
  - sha256 matches the manifest's sha256
  - reassembled bytes == source file bytes
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(".")
manifest = json.loads((ROOT / "www" / "weights.manifest.json").read_text())

src = (ROOT / "www" / "weights.json").read_bytes()
assert len(src) == manifest["total_bytes"], (
    f"manifest total_bytes {manifest['total_bytes']} != source {len(src)}"
)
assert hashlib.sha256(src).hexdigest() == manifest["sha256"], "sha256 mismatch"

template = manifest["filename_template"]
out = bytearray()
for i in range(1, manifest["chunk_count"] + 1):
    chunk_path = ROOT / "www" / (template % i)
    out.extend(chunk_path.read_bytes())

print(f"reassembled {len(out):,} bytes")
print(f"sha256 of reassembled: {hashlib.sha256(out).hexdigest()[:16]}...")
print(f"sha256 of source:      {hashlib.sha256(src).hexdigest()[:16]}...")
assert bytes(out) == src, "BYTE MISMATCH"
assert hashlib.sha256(out).hexdigest() == manifest["sha256"], "sha256 mismatch after reassemble"
print("OK: chunks reassemble byte-exact")