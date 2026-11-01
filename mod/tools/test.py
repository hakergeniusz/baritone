#!/usr/bin/env python3
"""The standalone suite: assemble the brain, then format, clippy and test it."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        sys.exit(result.returncode)


run([sys.executable, str(ROOT / "tools" / "assemble.py")])
# The runtime is generated: let rustfmt settle it, then hold it to the
# same standard as hand-written code.
run(["cargo", "fmt"])
run(["cargo", "fmt", "--check"])
run(["cargo", "clippy", "--all-targets", "--", "-D", "warnings"])
run(["cargo", "test", "--locked"])
print("baritone: standalone suite passed")
