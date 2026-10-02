"""Monitor the weather36 serial sweep without touching training or test data."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/weather36"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def parse_progress(text, names, state):
    """Only explicit DONE markers count as successful candidate completion."""
    result = {name: {"status": "NOT_STARTED"} for name in names}
    current = None
    epoch_pattern = re.compile(r"\[Epoch (\d+)/(\d+)\] train_loss=(\S+) val_loss=(\S+) lr=(\S+)")
    for line in text.splitlines():
        start = re.match(r"START candidate=(\d+)\b", line)
        done = re.match(r"DONE candidate=(\d+)\b", line)
        if start:
            current = names[int(start[1])]
            result[current] = {"status": "RUNNING", "epoch_history": []}
        elif done:
            result[names[int(done[1])]]["status"] = "COMPLETED"
            current = None
        elif current:
            match = epoch_pattern.search(line)
            if match:
                epoch, total, train, valid, lr = match.groups()
                entry = dict(epoch=int(epoch), total_epochs=int(total), train_loss=float(train),
                             val_loss=float(valid), lr=float(lr))
                result[current]["epoch_history"].append(entry)
                result[current]["latest_epoch"] = entry
    if current and state in TERMINAL:
        result[current]["status"] = "INCOMPLETE" if state == "COMPLETED" else state
    return result


def snapshot(folder, job, log_path=None):
    manifest = json.loads((folder / "tuning/manifest.json").read_text())
    state = "UNKNOWN"
    try:
        query = subprocess.run(["sacct", "-j", job, "--format=JobID,State,Elapsed,ExitCode", "-n", "-P"],
                               capture_output=True, text=True, timeout=30, check=True)
        scheduler = query.stdout.strip()
        for line in scheduler.splitlines():
            fields = line.split("|")
            if fields[0] == job:
                state = fields[1].split()[0].rstrip("+")
    except (OSError, subprocess.SubprocessError) as error:
        scheduler = str(error)
    log = Path(log_path) if log_path else folder / f"{job}-serial.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    age = time.time() - log.stat().st_mtime if log.exists() else None
    candidates = parse_progress(text, manifest["candidates"], state)
    if log_path:
        # Parallel worker logs cover only their assigned subset of the sweep.
        indices = {int(i) for i in re.findall(r"^START candidate=(\d+)\b", text, re.M)}
        candidates = {manifest["candidates"][i]: candidates[manifest["candidates"][i]]
                      for i in sorted(indices)}
    checkpoints = []
    for path in (folder / "runs").glob("*/checkpoints/**/*.ckpt"):
        stat = path.stat()
        checkpoints.append(dict(path=str(path), bytes=stat.st_size, modified=stat.st_mtime))
    alerts = []
    if state in TERMINAL - {"COMPLETED"}:
        alerts.append(f"Job stopped: {state}; inspect logs before resuming")
    if state == "RUNNING" and age is not None and age > 1800:
        alerts.append("No log update for 30 minutes")
    if state == "COMPLETED" and any(v["status"] != "COMPLETED" for v in candidates.values()):
        alerts.append("Scheduler completed but not all candidates have DONE markers")
    if re.search(r"(?:train_loss|val_loss)=(?:nan|[+-]?inf)\b", text, re.I):
        alerts.append("Nonfinite training/validation loss detected")
    return dict(time=dt.datetime.now().astimezone().isoformat(), job=job, state=state,
                scheduler=scheduler, log_age_seconds=age, candidates=candidates,
                checkpoints=checkpoints, alerts=alerts,
                scope="Training progress only; generation evaluation and model selection still pending; no test access.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT)
    parser.add_argument("--job", required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--hours", type=float, default=49)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.interval <= 0 or args.hours <= 0:
        parser.error("interval and hours must be positive")
    output = args.output or args.folder / "monitor"
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    while True:
        record = snapshot(args.folder, args.job, args.log)
        tmp = output / "latest.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(output / "latest.json")
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(record["time"], record["state"], record["alerts"], flush=True)
        if args.once or record["state"] in TERMINAL or time.monotonic() >= deadline:
            break
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()