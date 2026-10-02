"""Single-job continuation of a specific VerbalTS training run, failing closed."""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from sae.provenance import sha256_file
from sae.resolve_best_checkpoint import resolve_best_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/cpu_shape_v1'


def upstream_state(job):
    output = subprocess.check_output(['sacct', '-j', job, '-n', '-P',
        '--format=JobIDRaw,State,ExitCode'], text=True)
    for row in output.splitlines():
        fields = row.split('|')
        if fields[0] == job:
            state = fields[1].split()[0].rstrip('+')
            if state == 'COMPLETED' and fields[2] == '0:0':
                return True
            if state not in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
                raise RuntimeError(f'Upstream job failed: {row}')
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-job', required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def status(stage, **extra):
        value = dict(stage=stage, updated=datetime.datetime.now().astimezone().isoformat(), **extra)
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(out / 'status.json')
        print(value, flush=True)

    def run(stage, module, *argv):
        command = [sys.executable, '-u', '-m', module, *map(str, argv)]
        status(stage, command=command)
        with (out / f'{stage}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status('WAITING_VERBALTS', upstream_job=args.upstream_job)
        deadline = time.monotonic() + 48 * 3600
        while not upstream_state(args.upstream_job):
            if time.monotonic() > deadline:
                raise TimeoutError('VerbalTS did not finish in 48 hours')
            time.sleep(60)
        ckpt = resolve_best_checkpoint(args.experiment)
        validation = json.loads((DATA / 'validation.json').read_text())
        assert validation['status'] == 'PASS'
        for name, digest in validation['hashes'].items():
            assert sha256_file(DATA / name) == digest, name
        embedding_info = json.loads((ROOT / 'results/cpu_shape_v1/embedding_provenance.json').read_text())
        for split, info in embedding_info['splits'].items():
            assert sha256_file(DATA / f'{split}_cap_emb.npy') == info['embedding_sha256']
        (out / 'inputs.json').write_text(json.dumps(dict(checkpoint=str(ckpt),
            checkpoint_sha256=sha256_file(ckpt), dataset=validation), indent=2))
        run('CNN', 'scripts.train_cpu_pipeline_cnn', '--data-root', DATA, '--output-dir', out / 'cnn')
        for split in ('train', 'valid'):
            run(f'ACTIVATIONS_{split}', 'sae.activations', '--ckpt-path', ckpt, '--data-root', DATA,
                '--split', split, '--target-t-ranges', '5-45', '--output-dir', out / 'activations',
                '--batch-size', 64, '--device', 'cuda')
        train_cache = out / 'activations/train_layer1_t5_45.pt'
        valid_cache = out / 'activations/valid_layer1_t5_45.pt'
        run('SAE', 'sae.training.train_sae', '--activations', train_cache,
            '--valid-activations', valid_cache, '--output-dir', out / 'sae', '--t-range', 5, 45,
            '--input-dim', 64, '--latent-dim', 512, '--topk-k', 48, '--epochs', 100,
            '--batch-size', 4096, '--device', 'cuda')
        run('MLP', 'sae.train_classifier', '--sae-checkpoint', out / 'sae/best.pt',
            '--train-cache', train_cache, '--valid-cache', valid_cache,
            '--train-attrs', DATA / 'train_attrs_idx.npy', '--valid-attrs', DATA / 'valid_attrs_idx.npy',
            '--num-heads', 3, '--selection-metric', 'accr', '--output-dir', out / 'mlp',
            '--epochs', 100, '--patience', 12, '--device', 'cuda')
        for seed in (1, 7, 42):
            run(f'TEST_ACTIVATIONS_{seed}', 'sae.activations', '--ckpt-path', ckpt,
                '--data-root', DATA, '--split', 'test', '--target-t-ranges', '5-45',
                '--output-dir', out / f'test_activations_{seed}', '--seed', seed,
                '--batch-size', 64, '--device', 'cuda')
        run('EVALUATION', 'scripts.eval_cpu_pipeline', '--data-root', DATA,
            '--output-dir', out, '--checkpoint', ckpt)
        status('COMPLETE', report=str(out / 'report.md'))
    except BaseException as error:
        previous = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        status('FAILED', failed_stage=previous.get('stage'), error=repr(error))
        raise


if __name__ == '__main__':
    main()