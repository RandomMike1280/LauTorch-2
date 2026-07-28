"""update_chunks.py — Update chunk files from the current www/weights.json.

Idempotent: wipes old chunks, splits the source, writes the manifest, and
verifies byte-exact reassembly. No arguments needed — all paths are hardcoded
to match the LauTorch www/ layout.

Usage:
    python tools/update_chunks.py
"""
import hashlib
import json
import sys
from pathlib import Path

# Hardcoded layout — architecture is fixed, so these never change.
WWW_DIR   = Path("www")
SRC_FILE  = WWW_DIR / "weights.json"
MANIFEST  = WWW_DIR / "weights.manifest.json"
BASE_NAME = "weights"
CHUNK_SIZE = 2000          # bytes; must stay < 3000 for Lau's string cap
TEMPLATE  = f"{BASE_NAME}.part%03d.json"  # 3-digit zero-padded, 1-based


def main() -> int:
    print("=== LauTorch Chunk Updater ===\n")

    if not SRC_FILE.exists():
        print(f"ERROR: source not found: {SRC_FILE}", file=sys.stderr)
        return 1

    raw = SRC_FILE.read_bytes()
    total = len(raw)
    new_digest = hashlib.sha256(raw).hexdigest()

    # Show previous SHA if manifest exists so we know if weights changed.
    prev_digest = None
    if MANIFEST.exists():
        try:
            prev = json.loads(MANIFEST.read_text())
            prev_digest = prev.get("sha256")
        except Exception:
            pass

    print(f"Source:  {SRC_FILE}  ({total:,} bytes)")
    if prev_digest:
        print(f"Previous SHA-256: {prev_digest}")
    print(f"New SHA-256:      {new_digest}\n")

    if prev_digest == new_digest:
        print("SHA-256 unchanged — nothing to do.")
        return 0

    print("Weights changed — re-chunking...\n")

    WWW_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe old chunks.
    for old in WWW_DIR.glob(f"{BASE_NAME}.part*.json"):
        old.unlink()

    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Chunks written: {n_chunks} x {CHUNK_SIZE} bytes (last: {total - (n_chunks - 1) * CHUNK_SIZE} bytes)")

    # Write chunks.
    for i in range(1, n_chunks + 1):
        start = (i - 1) * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, total)
        chunk_path = WWW_DIR / (TEMPLATE % i)
        chunk_path.write_bytes(raw[start:end])

    # Write manifest.
    manifest = {
        "chunk_count":       n_chunks,
        "chunk_size":        CHUNK_SIZE,
        "total_bytes":       total,
        "sha256":            new_digest,
        "filename_template": TEMPLATE,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest:        {MANIFEST}\n")

    # Verify byte-exact reassembly.
    print("Verifying byte-exact reassembly...")
    reassembled = bytearray()
    for i in range(1, n_chunks + 1):
        reassembled.extend((WWW_DIR / (TEMPLATE % i)).read_bytes())

    assembled_digest = hashlib.sha256(reassembled).hexdigest()
    print(f"  sha256 reassembled: {assembled_digest}")
    print(f"  sha256 source:      {new_digest}")

    if assembled_digest != new_digest:
        print("\nERROR: reassembly mismatch — chunks are corrupted!", file=sys.stderr)
        return 1

    print("OK: chunks reassemble byte-exact.\n")
    print("=== Done. Commit the following files ===")
    print("  www/weights.json")
    print("  www/weights.manifest.json")
    print(f"  www/{BASE_NAME}.part*.json  ({n_chunks} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
