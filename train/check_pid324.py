import subprocess
out = subprocess.run(["tasklist", "/FI", "PID eq 3208"], capture_output=True, text=True)
print(out.stdout)
print("stderr:", out.stderr)
