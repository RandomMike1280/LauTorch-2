import os
import subprocess
out = subprocess.run(["tasklist", "/FI", "PID eq 3056"], capture_output=True, text=True)
print(out.stdout)
print(out.stderr)
