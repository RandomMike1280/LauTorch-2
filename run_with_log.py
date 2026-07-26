"""Run chat.lau via the Interpreter API, capture stdout, write debug log."""
import sys
import json
import os
from datetime import datetime
from lau import Interpreter, RuntimeConfig

LOG_PATH = r"C:\Users\angel\OneDrive\Desktop\LauTorch-2\debug2.log"
SCRIPT = r"C:\Users\angel\OneDrive\Desktop\LauTorch-2\chat.lau"

# Clear log file
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

# Capture stdout to both terminal and log
class TeeOutput:
    def __init__(self, log_path):
        self.log = open(log_path, 'a', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self._stdout.flush()
        # Also write to log
        if s.strip():
            self.log.write(s)
            if not s.endswith('\n'):
                self.log.write('\n')
            self.log.flush()
    def flush(self):
        self._stdout.flush()

sys.stdout = TeeOutput(LOG_PATH)

config = RuntimeConfig(realtime=False)
interp = Interpreter(config)
result = interp.run_file(SCRIPT)
print(f"\n[DEBUG] exit_code={result.exit_code}, success={result.success}")
print(f"[DEBUG] stdout=\n{result.stdout}")
print(f"[DEBUG] stderr=\n{result.stderr}")
if result.error:
    print(f"[DEBUG] error: {result.error}")
sys.stdout = sys.stdout._stdout