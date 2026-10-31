"""Prepare the published evidence and launch the original GridPulse dashboard."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.prepare_release import prepare


def main() -> None:
    root = Path(__file__).resolve().parent
    print("Verifying bundled V8 evidence...", flush=True)
    result = prepare(root)
    print(f"Evidence ready: {result['files_verified']} verified files.", flush=True)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(root / "app.py"), *sys.argv[1:]],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
