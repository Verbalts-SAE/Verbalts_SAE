"""Complete the retained seed-147 protocol without selecting new configurations."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import textwrap

import numpy as np

ROOT = Path('/public/home/liym2024/Verbalts_SAE')
BENCH = Path('/public/home/liym2024/ConTSG-Bench')
RESULTS = ROOT / 'results/electricity_v3'
OUT = RESULTS / 'seed147_complete'
DATA = ROOT / 'datasets/electricity_15min_semisynth_morph'
SOURCES = {'Bridge': ('bridge', '1012167', 'i2u_gs_raw'),
           'DiffuSETS': ('diffusets', '1012168', 't8_i2')}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    for model, (family, job, variant) in SOURCES.items():
        source = RESULTS / f'steering/{family}_stageD_{job}'
        old = read(source / 'seed1/baseline/summary.json')
        for name, key in [('test_text_caps.npy', 'captions_sha256'),
                          ('test_cap_emb.npy', 'embedding_sha256'),
                          ('test_ts.npy', 'references_sha256')]:
            assert digest(DATA / name) == old['audit'][key], name
        ckpt = Path(old['audit']['model']['checkpoint'])
        assert digest(ckpt) == old['audit']['model']['checkpoint_sha256']
        config = RESULTS / f'steering_infra/configs/{family}_electricity_v3_clean.yaml'
        assert digest(config) == old['pairing_fingerprint']['config_sha256']
        dest = OUT / family
        dest.mkdir(exist_ok=True)
        for seed in (1, 42):
            link = dest / f'seed{seed}'
            if not link.exists():
                link.symlink_to(source / f'seed{seed}', target_is_directory=True)
        if (dest / f'seed7/{variant}/summary.json').exists():
            continue
        script = (ROOT / f'scripts/eval_{family}_stageD.slurm').read_text()
        script = script.split('# ---- summary + CTTP ----')[0]
        script = '\n'.join(line for line in script.splitlines()
                           if not line.startswith('#SBATCH') and not line.startswith('TRAINDIR='))
        script = script.replace('CKPT="$TRAINDIR/checkpoints/finetune/last.ckpt"', f'CKPT="{ckpt}"')
        script = script.replace(f'OUT=$ROOT/results/electricity_v3/steering/{family}_stageD_$SLURM_JOB_ID', f'OUT="{dest}"')
        script = script.replace('for SEED in 1 11 42;', 'for SEED in 7;')
        path = OUT / f'run_{family}_seed7.sh'
        path.write_text(script + '\n')
        subprocess.run(['bash', '-n', str(path)], check=True)
        subprocess.run(['bash', str(path)], check=True)


def report():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sae.evaluate_bridge_steering import _load_evaluation_classifier
    from sae.evaluate_diffusets_steering import classify_shape_heads
    from sae.visualize_variants import captions_to_targets, parse_segment_shapes

    OUT.mkdir(parents=True, exist_ok=True)
    cnn = RESULTS / 'cnn/segment_cnn.pth'
    evaluator = _load_evaluation_classifier(cnn, torch.device('cuda'))
    caps = np.load(DATA / 'test_text_caps.npy', allow_pickle=True).reshape(-1)
    truth = np.load(DATA / 'test_ts.npy').reshape(2000, 128)
    rows, records, cases = [], {}, []
    for model in ('Bridge', 'DiffuSETS', 'VerbalTS'):
        metrics = []
        for seed in (1, 7, 42):
            if model == 'VerbalTS':
                folder = RESULTS / f'priority_test_1011172/seed{seed}'
                summary = read(folder / 'summary.json')
                indices = summary['indices']
                paths = [folder / 'pure.npy', folder / 'previous_reference.npy']
                assert digest(DATA / 'test_text_caps.npy') == summary['data_sha256']['test_text_caps.npy']
                assert digest(cnn) in summary['checkpoints'].values()
            else:
                family, _, variant = SOURCES[model]
                folder = OUT / family / f'seed{seed}'
                paths = [folder / 'baseline/curves.npy', folder / variant / 'curves.npy']
                indices = list(range(2000))
                old = read(RESULTS / f'steering/{family}_stageD_{SOURCES[model][1]}/seed1/baseline/summary.json')
                for name in ('baseline', variant):
                    current = read(folder / name / 'summary.json')
                    for key in ('captions_sha256', 'embedding_sha256', 'evaluation_classifier_sha256', 'references_sha256'):
                        assert current['audit'][key] == old['audit'][key], (model, seed, key)
                    assert current['pairing_fingerprint']['generator_checkpoint_sha256'] == old['pairing_fingerprint']['generator_checkpoint_sha256']
            assert sorted(indices) == list(range(2000))
            work = OUT / 'scores' / model / f'seed{seed}'
            work.mkdir(parents=True, exist_ok=True)
            indexfile = work / 'indices.json'
            indexfile.write_text(json.dumps(indices))
            curves = [np.load(p).reshape(2000, 128) for p in paths]
            _, peak, valley = captions_to_targets(caps[indices])
            matches, values = [], []
            for label, path, curve in zip(('baseline', 'steering'), paths, curves):
                assert np.isfinite(curve).all()
                _, match, _ = classify_shape_heads(evaluator, curve, peak, valley, torch.device('cuda'))
                matches.append(match)
                score = work / f'cttp_{label}.json'
                if not score.exists():
                    subprocess.run([sys.executable, str(ROOT / 'scripts/score_curves_cttp.py'),
                                    '--curves', str(path), '--captions', str(DATA / 'test_text_caps.npy'),
                                    '--indices-json', str(indexfile), '--output', str(score)], check=True)
                cttp = read(score)
                assert cttp['n_samples'] == 2000 and cttp['curves_sha256'] == digest(path)
                values.append([float(np.mean((curve - truth[indices]) ** 2)),
                               float(match.all(axis=1).mean()), cttp['cttp_mean']])
            metrics.append(values)
            if seed == 1:
                fixed = np.flatnonzero(~matches[0].all(axis=1) & matches[1].all(axis=1))
                # Deterministic selection: first dataset IDs with distinct target triples.
                seen, selected = set(), []
                for pos in sorted(fixed, key=lambda p: indices[p]):
                    target = parse_segment_shapes(str(caps[indices[pos]]))
                    if target in seen:
                        continue
                    seen.add(target)
                    selected.append(int(pos))
                    if len(selected) == 4:
                        break
                assert len(selected) == 4, (model, len(fixed))
                for number, pos in enumerate(selected, 1):
                    idx = indices[pos]
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(truth[idx], color='0.5', linestyle='--', label='Ground truth', alpha=.7)
                    ax.plot(curves[0][pos], color='#D55E00', label='Baseline')
                    ax.plot(curves[1][pos], color='#0072B2', label='Steering')
                    for boundary in (43, 85):
                        ax.axvline(boundary, color='0.7', linestyle=':')
                    target = parse_segment_shapes(str(caps[idx]))
                    ax.set_title(f'{model} | seed 1 | test ID {idx} | FIXED: baseline fail -> steering pass\n'
                                 + ' / '.join(target), fontsize=11)
                    ax.set_xlabel('Time step'); ax.set_ylabel('Value (original evaluation scale)')
                    ax.legend(); ax.grid(alpha=.15)
                    fig.text(.05, .02, textwrap.fill(str(caps[idx]), 145), fontsize=8, va='bottom')
                    fig.subplots_adjust(bottom=.32)
                    name = f'{model}_fixed_{number}_id{idx}'
                    for ext in ('png', 'pdf'):
                        fig.savefig(OUT / f'{name}.{ext}', dpi=160)
                    plt.close(fig)
                    cases.append({'model': model, 'seed': 1, 'test_id': idx, 'row': pos,
                                  'caption': str(caps[idx]), 'target': target,
                                  'baseline_segment_correct': matches[0][pos].tolist(),
                                  'steering_segment_correct': matches[1][pos].tolist(), 'figure': name})
        array = np.asarray(metrics)
        records[model] = {'seeds': [1, 7, 42], 'metrics': ['mse', 'accr', 'cttp'], 'values': array.tolist()}
        for i, metric in enumerate(('mse', 'accr', 'cttp')):
            base, steer = array[:, 0, i], array[:, 1, i]
            delta = (steer - base) * (100 if metric == 'accr' else 1)
            unit = ' pp' if metric == 'accr' else ''
            rows.append([model, metric, f'{base.mean():.5f} ± {base.std(ddof=1):.5f}',
                         f'{steer.mean():.5f} ± {steer.std(ddof=1):.5f}',
                         f'{delta.mean():+.5f} ± {delta.std(ddof=1):.5f}{unit}'])
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2))
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=2))
    header = ['model', 'index', 'baseline', 'steering', 'change']
    text = '# Result (seed 1,7,42)\n\nNew full captions; test N=2000; sample std (ddof=1); paired changes.\n\n'
    text += '| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---'] * 5) + '|\n'
    text += '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
    text += '\n\nVerbalTS steering = previous_reference (screenshot variant, not reselected on test).\n'
    text += 'Cases are selectively illustrated CNN-defined fixes, not representative population estimates.\n'
    text += 'A CNN pass does not guarantee visual correctness or accurate amplitudes (notably DiffuSETS ID 7).\n'
    text += 'The test set was viewed in earlier experiments; these are retained-configuration descriptive results, not an untouched confirmatory test.\n'
    text += 'Cross-seed standard deviations measure generation randomness, not generalization confidence intervals.\n'
    text += 'Validation-only configuration-selection history and exact historical dirty source snapshots remain unverified.\n'
    text += 'Case rule: seed 1, ascending test ID, distinct target triples; first four fixed cases per model.\n'
    text += '\n' + '\n'.join(f"![{c['model']} ID {c['test_id']}]({c['figure']}.png)" for c in cases)
    (OUT / 'REPORT.md').write_text(text)
    fig, ax = plt.subplots(figsize=(13, 5)); ax.axis('off')
    ax.set_title('Result (seed 1,7,42) — new full captions, test N=2000', pad=15)
    table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2)
    fig.tight_layout(); fig.savefig(OUT / 'result_table.png', dpi=200); plt.close(fig)
    print(text.split('Cases are')[0], flush=True)


if __name__ == '__main__':
    if '--report-only' not in sys.argv:
        generate()
    report()