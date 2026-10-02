"""Read-only workload monitoring; records snapshots without selecting on test."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot(folder: Path, job: str) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    states = {}
    for candidate in manifest["candidates"]:
        path = folder / "runs" / candidate["name"] / "status.txt"
        states[candidate["name"]] = path.read_text().strip() if path.exists() else "NOT_STARTED"
    try:
        result = subprocess.run(
            ["sacct", "-j", job, "--format=JobID,State,Elapsed,ExitCode", "-n", "-P"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        scheduler = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        scheduler = f"MONITOR_QUERY_ERROR: {exc}"
    log = ROOT / "results/electricity_v3/logs" / f"{job}-three-serial.log"
    tail = ""
    age = None
    if log.exists():
        age = time.time() - log.stat().st_mtime
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 16384))
            tail = "\n".join(handle.read().decode(errors="replace").splitlines()[-15:])
    return {
        "time": dt.datetime.now().astimezone().isoformat(), "job": job,
        "candidate_states": states, "scheduler": scheduler,
        "log_age_seconds": age, "log_tail": tail,
        "attention": any(s == "FAILED" for s in states.values())
        or (age is not None and age > 1800 and "RUNNING" in scheduler),
        "scope": "Monitoring only. Pilot completion is not final model selection. No automatic repair.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.hours <= 0:
        parser.error("hours and interval must be positive")
    output = args.folder / "monitor"
    output.mkdir(exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    while True:
        record = snapshot(args.folder, args.job)
        temporary = output / "latest.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output / "latest.json")
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(record["time"], record["candidate_states"], flush=True)
        if args.once or time.monotonic() >= deadline:
            break
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()