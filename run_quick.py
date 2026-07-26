"""Run a Lau script and write to file, time-limited."""
import sys
import os
import threading
import time

SCRIPT = r"C:\Users\angel\OneDrive\Desktop\LauTorch-2\test_simple.lau"
OUT = r"C:\Users\angel\OneDrive\Desktop\LauTorch-2\out.txt"

# Open file in unbuffered mode
outf = open(OUT, 'w', buffering=1)
outf.write("START\n")
outf.flush()

# Use the lau interpreter
from lau import Interpreter, RuntimeConfig

config = RuntimeConfig(realtime=False)
interp = Interpreter(config)

# Set max execution time (in seconds) - use a small limit
outf.write("Loading script...\n")
outf.flush()
try:
    result = interp.run_file(SCRIPT)
    outf.write(f"\n[DEBUG] exit_code={result.exit_code}, success={result.success}\n")
    outf.write(f"[DEBUG] stdout=\n{result.stdout}\n")
    if result.error:
        outf.write(f"[DEBUG] error: {result.error}\n")
except Exception as e:
    import traceback
    outf.write(f"\n[DEBUG] EXCEPTION: {e}\n")
    outf.write(traceback.format_exc())
outf.write("END\n")
outf.flush()
outf.close()
print("WROTE TO FILE")
