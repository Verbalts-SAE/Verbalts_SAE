"""Persist scheduler and pipeline progress, including timeout/cancellation."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

from scripts.run_weather128_difficulty import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    deadline = time.monotonic() + 26 * 3600
    while time.monotonic() < deadline:
        record = dict(time=dt.datetime.now().astimezone().isoformat(), job=args.job, state='UNKNOWN')
        try:
            query = subprocess.run(['sacct', '-j', args.job, '--format=JobID,State,Elapsed,ExitCode', '-n', '-P'],
                                   capture_output=True, text=True, timeout=30, check=True)
            record['scheduler'] = query.stdout
            for line in query.stdout.splitlines():
                fields = line.split('|')
                if fields[0] == args.job:
                    record['state'] = fields[1].split()[0].rstrip('+')
            folder = args.root / f'run_{args.job}'
            for name in ('status', 'monitor'):
                path = folder / f'{name}.json'
                if path.exists():
                    record[name] = json.loads(path.read_text())
            stage_log = Path(record.get('status', {}).get('log', args.root / f'{args.job}-pipeline.log'))
            if stage_log.exists():
                record['log_age_seconds'] = time.time() - stage_log.stat().st_mtime
                record['tail'] = stage_log.read_text(errors='replace')[-4000:]
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record['error'] = repr(exc)
        save(args.root / f'{args.job}-latest.json', record)
        with (args.root / f'{args.job}-history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(record['time'], record['state'], record.get('status', {}).get('status'), flush=True)
        if record['state'] in terminal:
            break
        time.sleep(300)


if __name__ == '__main__':
    main()