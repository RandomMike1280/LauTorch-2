"""Run a Lau script with timeout."""
import sys
from lau import Interpreter, RuntimeConfig

SCRIPT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\angel\OneDrive\Desktop\LauTorch-2\test_lau_one.lau"

config = RuntimeConfig(realtime=False)
interp = Interpreter(config)
result = interp.run_file(SCRIPT)
print(f"\n[DEBUG] exit_code={result.exit_code}, success={result.success}")
print(f"[DEBUG] stdout=\n{result.stdout}")
print(f"[DEBUG] stderr=\n{result.stderr}")
if result.error:
    print(f"[DEBUG] error: {result.error}")
