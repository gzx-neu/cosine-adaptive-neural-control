"""Launch the remaining frozen suite as a detached, no-window process."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "kkt_collocation/results/unified_su_suj_sk_konly_20seeds_v1"


def main() -> None:
    seeds = [str(seed) for seed in range(20260772, 20260791)]
    command = [
        sys.executable,
        "kkt_collocation/run_unified_20seed_suite.py",
        "--seeds",
        *seeds,
        "--benchmarks",
        "vdp",
        "penicillin",
        "cstr",
        "--workers",
        "4",
    ]
    stdout_path = RESULT / "suite_background_stdout.log"
    stderr_path = RESULT / "suite_background_stderr.log"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    launch = {
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "seeds": seeds,
        "no_window": sys.platform == "win32",
    }
    (RESULT / "background_launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
