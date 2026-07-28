"""chunk_weights.py — split www/weights.json into chunks Lau can fetch.

Reads www/weights.json and emits:
  - www/weights.manifest.json
  - www/weights.part000.json
  - www/weights.part001.json
  - ...

Manifest schema (matches what download_weights.laum expects):
  {
    "chunk_count": int,           # number of chunk files
    "chunk_size":  int,           # bytes per chunk (last may be shorter)
    "total_bytes": int,           # total bytes of the original file
    "sha256":      str,           # hex digest of original file for verification
    "filename_template": str,     # e.g. "weights.part%03d.json"
  }

Re-running this script is idempotent: it deletes any existing chunk files
before writing new ones, but leaves weights.json itself untouched.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="www/weights.json",
        help="Path to the source weights JSON (default: www/weights.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Bytes per chunk (default: 2000; must be < 3000 to fit Lau's string cap)",
    )
    parser.add_argument(
        "--out-dir",
        default="www",
        help="Directory to write manifest + chunk files into (default: www)",
    )
    parser.add_argument(
        "--manifest-name",
        default="weights.manifest.json",
        help="Manifest filename (default: weights.manifest.json)",
    )
    parser.add_argument(
        "--base-name",
        default="weights",
        help="Base name for chunk files (default: weights)",
    )
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = src.read_bytes()
    total = len(raw)
    digest = hashlib.sha256(raw).hexdigest()

    chunk_size = args.chunk_size
    if chunk_size <= 0:
        print("ERROR: chunk-size must be > 0", file=sys.stderr)
        return 1
    if chunk_size >= 3000:
        print(
            f"WARNING: chunk-size {chunk_size} >= 3000 — risks hitting Lau's "
            "per-string cap. Continuing anyway because user asked for it.",
            file=sys.stderr,
        )

    # Wipe any existing chunk files so re-runs don't leave stale debris.
    for old in out_dir.glob(f"{args.base_name}.part*.json"):
        old.unlink()

    n_chunks = (total + chunk_size - 1) // chunk_size
    pad = max(3, len(str(n_chunks)))
    template = f"{args.base_name}.part%0{pad}d.json"

    for i in range(1, n_chunks + 1):  # 1-indexed to match Lau's loop convention
        start = (i - 1) * chunk_size
        end = min(start + chunk_size, total)
        chunk_path = out_dir / (template % i)
        chunk_path.write_bytes(raw[start:end])

    manifest = {
        "chunk_count": n_chunks,
        "chunk_size": chunk_size,
        "total_bytes": total,
        "sha256": digest,
        "filename_template": template,
    }
    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Source: {src} ({total:,} bytes, sha256={digest[:16]}...)")
    print(f"Chunks: {n_chunks} x {chunk_size} bytes (last may be shorter)")
    print(f"Manifest: {manifest_path}")
    print(f"Pattern: {template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())