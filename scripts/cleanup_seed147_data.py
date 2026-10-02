"""Conservative, audited cleanup of superseded experiment payloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/public/home/liym2024/Verbalts_SAE')
OUT = ROOT / 'results/electricity_v3/seed147_complete'
AUDIT = OUT / 'cleanup_audit'
KEEP = [OUT, ROOT / 'datasets/electricity_15min_semisynth_morph']
KEEP += [ROOT / 'results/electricity_v3' / name for name in (
    'priority_test_1011172', 'steering/bridge_stageD_1012167',
    'steering/diffusets_stageD_1012168', 'steering_infra',
    'stabilized_global_soft_1010876', 'activations',
)]


def under(path, parent):
    return path == parent or parent in path.parents


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=str(ROOT)).decode().split('\0'))
    protected = set()
    # Retain every symlink target, including targets outside the final report.
    for top in ('results', 'datasets', 'outputs', 'checkpoints'):
        for p in (ROOT / top).rglob('*'):
            if p.is_symlink():
                protected.add(p.resolve())
    # Retain explicit local paths recorded in metadata and source/config files.
    import re
    for top in ('scripts', 'configs', 'sae', 'results'):
        for p in (ROOT / top).rglob('*'):
            if under(p, AUDIT):
                continue
            if p.suffix in ('.json', '.yaml', '.yml', '.py', '.sh', '.slurm') and p.is_file():
                for value in re.findall(r'/public/home/liym2024/Verbalts_SAE/[^\s\"\'<>;,\)]+', p.read_text(errors='replace')):
                    q = Path(value)
                    if q.is_file():
                        protected.add(q.resolve())
    candidates = []
    for top in ('results', 'datasets', 'outputs'):
        for p in (ROOT / top).rglob('*'):
            if not p.is_file() or p.is_symlink() or str(p.relative_to(ROOT)) in tracked:
                continue
            if any(under(p.resolve(), q) for q in protected):
                continue
            old_caption = top == 'datasets' and '.prev_' in p.name and p.suffix == '.npy'
            obsolete_caption = (top == 'datasets' and not under(p, KEEP[1])
                                and ('text_caps' in p.name or 'cap_emb' in p.name)
                                and p.suffix == '.npy')
            payload = (top == 'results' and p.suffix.lower() in
                       ('.npy', '.npz', '.png', '.pdf', '.jpg', '.jpeg')
                       and not any(under(p, q) for q in KEEP)
                       and not any(x in p.parts for x in ('checkpoints', 'models', 'cnn', 'classwise_classifiers')))
            if old_caption or obsolete_caption or payload:
                candidates.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    AUDIT.mkdir(exist_ok=True)
    manifest = {'policy': 'Keep raw time series, all weights, code, metadata, dependencies and symlink targets; remove only obsolete captions and unrelated result payloads.',
                'files': candidates, 'bytes': sum(x['bytes'] for x in candidates)}
    (AUDIT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Candidates:', len(candidates), 'bytes:', manifest['bytes'], flush=True)
    if '--apply' not in sys.argv:
        return
    preserved = {}
    for parent in KEEP:
        for p in parent.rglob('*'):
            if p.is_file() and not under(p, AUDIT) and '.prev_' not in p.name:
                preserved[str(p)] = sha(p)
    (AUDIT / 'preserved_sha256.json').write_text(json.dumps(preserved, indent=2))
    deleted = []
    for row in candidates:
        p = Path(row['path'])
        assert sha(p) == row['sha256'], p
        p.unlink()
        deleted.append(row['path'])
        (AUDIT / 'deleted.json').write_text(json.dumps(deleted, indent=2))
    for name, expected in preserved.items():
        assert sha(Path(name)) == expected, name
    for p in OUT.rglob('*'):
        if p.is_symlink():
            assert p.exists(), p
    status = {'deleted_files': len(deleted), 'deleted_bytes': manifest['bytes'],
              'unchanged_protected_files': len(preserved), 'status': 'PASS'}
    (AUDIT / 'verification.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()