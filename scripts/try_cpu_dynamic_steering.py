"""Try the electricity_v3 dynamic-steering champion recipe on cpu_shape_v1.

Round 1 on cpu_shape_v1 (steering_1018520) exhausted topk in {16,48},
eta in {320,1280,5120}, gamma in {0,3}, windows {(5,45),(5,25)} with
max_step=.5, rel_cap=1., active_only=False: every candidate scored at or
below pure ACCR (weak doses were no-ops, strong doses were negative).

The electricity_v3 round-2 breakthrough was active_only=True (the per-token
top-k budget skips dead latents): ao_g0_e10240 with topk=32, eta=10240,
gamma=0, window=[5,45], max_step=1., rel_cap=2., selection_score=applied,
preserve_residual=True gained +4.65 pp there (pure 46.55% -> 51.20%).

This script screens that recipe family plus single-axis controls on the same
validation window as the round-1 screen (256 samples, seed 42), so numbers
are directly comparable with steering_1018520. Refine/confirm/test run only
when the screen produces an eligible winner.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import load_classifier_checkpoint
from sae.steering import LatentClassifierGuidanceWrapper, TimestepSAEWrapper, load_sae_checkpoint
from scripts.cpu_pipeline_metrics import accuracy_report, predict_cnn

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = dict(name='ref_e5120', topk=32, eta=5120, gamma=0., active_only=False,
                 guidance_t_range=[5, 45], max_step=1., rel_cap=2., iters=1)
CHAMPION = dict(REFERENCE, name='ao_e10240', active_only=True, eta=10240)
AUDIT_KEYS = ('guidance_applied_nonzero_mean_abs_step',
              'guidance_applied_mean_abs_step_over_sigma',
              'guidance_applied_nonzero_fraction',
              'guidance_cap_fraction', 'guidance_rows')


def candidates():
    pool = [REFERENCE, CHAMPION]

    def add(name, **overrides):
        pool.append(dict(REFERENCE, name=name, **overrides))

    # electricity_v3 champion family around the active_only axis.
    add('ao_e5120', active_only=True)
    add('ao_e20480', active_only=True, eta=20480)
    # Single-axis controls separating active_only from eta.
    add('ref_e10240', eta=10240)
    # Axes that were negative or neutral in electricity_v3, re-checked here.
    add('ao_e10240_g3', active_only=True, eta=10240, gamma=3.)
    add('ao_e10240_g6', active_only=True, eta=10240, gamma=6.)
    add('ao_e10240_k16', active_only=True, eta=10240, topk=16)
    add('ao_e10240_k48', active_only=True, eta=10240, topk=48)
    add('ao_e10240_i2', active_only=True, eta=10240, iters=2)
    add('ao_e10240_w5_25', active_only=True, eta=10240, guidance_t_range=[5, 25])
    add('ao_e10240_ms05', active_only=True, eta=10240, max_step=.5, rel_cap=1.)
    # Round-1 style legacy point kept as a protocol anchor (expected gain<=0).
    add('legacy_best', topk=16, eta=320, guidance_t_range=[5, 25], max_step=.5, rel_cap=1.)
    return pool


def rank(reports, configs):
    rows = []
    for config in configs:
        values = [r[config['name']] for r in reports]
        safe = all(all(v[key] <= r['pure'][key] * 1.10 + 1e-8
                       for key in ('mse', 'roughness', 'curvature', 'amplitude_p99'))
                   for v, r in zip(values, reports))
        gain = float(np.mean([v['accr'] - r['pure']['accr']
                              for v, r in zip(values, reports)]))
        rows.append(dict(config=config, eligible=safe and gain > 0, gain=gain))
    return sorted(rows, key=lambda r: (not r['eligible'], -r['gain'], r['config']['name']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true',
                   help='8 samples, reference + champion only, no refine')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data, base = ROOT / 'datasets/cpu_shape_v1', ROOT / 'results/cpu_shape_v1/pipeline'
    inputs = json.loads((base / 'inputs.json').read_text())
    checkpoint = Path(inputs['checkpoint'])
    assert sha256_file(checkpoint) == inputs['checkpoint_sha256']
    device = torch.device(args.device)
    model, _ = load_verbalts(checkpoint, device)
    sae, window, _ = load_sae_checkpoint(base / 'sae/best.pt', device)
    mlp, meta = load_classifier_checkpoint(base / 'mlp/best.pt', device)
    assert meta['sae_checkpoint_sha256'] == sha256_file(base / 'sae/best.pt')
    cnn = PeakValleyClassifier1D(segment_len=43).to(device)
    cnn.load_state_dict(torch.load(base / 'cnn/best.pt', map_location=device, weights_only=True))
    for net in (model, sae, mlp, cnn):
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    pool = [REFERENCE, CHAMPION] if args.smoke else candidates()
    write_json(out / 'protocol.json', dict(candidates=pool, batch_size=16,
        source='electricity_v3 ao champion family transferred to cpu_shape_v1',
        guardrail='Each seed: MSE, roughness, curvature, p99 amplitude <= pure * 1.10',
        caveat='CNN measures local morphology, not background caption compliance.',
        checkpoints={str(f): sha256_file(f) for f in
                     (checkpoint, base / 'sae/best.pt', base / 'mlp/best.pt', base / 'cnn/best.pt')}))

    def evaluate(stage, split, indices, seed, configs):
        write_json(out / 'status.json', dict(stage=stage, seed=seed, state='running'))
        folder = out / f'{stage}_{seed}'
        folder.mkdir()
        labels = np.load(data / f'{split}_attrs_idx.npy')[indices]
        truth = np.load(data / f'{split}_ts.npy')[indices].reshape(len(indices), 128)
        embeddings = np.load(data / f'{split}_cap_emb.npy')[indices].astype(np.float32)
        report = {}
        for config in [dict(name='pure'), dict(name='sae')] + configs:
            name = config['name']
            wrapper = None
            if name == 'sae':
                wrapper = TimestepSAEWrapper(sae, window)
            elif name != 'pure':
                wrapper = LatentClassifierGuidanceWrapper(sae, window, mlp,
                    eta=config['eta'], adaptive_gamma=config['gamma'], topk=config['topk'],
                    topk_mode='dynamic', active_only=config['active_only'],
                    guidance_t_range=config['guidance_t_range'], rel_cap=config['rel_cap'],
                    max_step=config['max_step'], iters=config['iters'],
                    selection_score='applied', preserve_residual=True)
            if wrapper is not None:
                wrapper.to(device).eval()
            parts = []
            for start in range(0, len(indices), 16):
                if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                    wrapper.set_targets(torch.as_tensor(labels[start:start + 16], device=device).long())
                with torch.no_grad():
                    parts.append(generate_variant(model, torch.as_tensor(
                        embeddings[start:start + 16], device=device), wrapper, seed + start))
            curves = np.concatenate(parts)
            assert curves.shape == truth.shape and np.isfinite(curves).all()
            predicted = predict_cnn(cnn, curves, device)
            np.savez(folder / f'{name}.npz', curves=curves, predicted=predicted,
                     labels=labels, indices=indices)
            entry = dict(**accuracy_report(predicted, labels),
                mse=float(np.mean((curves - truth) ** 2)),
                roughness=float(np.abs(np.diff(curves)).mean()),
                curvature=float(np.abs(np.diff(curves, n=2)).mean()),
                amplitude_p99=float(np.quantile(np.abs(curves).max(1), .99)))
            if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                audit = wrapper.audit()
                entry['dose'] = {key: audit[key] for key in AUDIT_KEYS}
            report[name] = entry
            write_json(folder / 'metrics.json', report)
            dose = entry.get('dose', {})
            print(stage, seed, name, round(entry['accr'], 6), 'mse', round(entry['mse'], 6),
                  'step', round(dose.get('guidance_applied_nonzero_mean_abs_step', 0.), 6),
                  'over_sigma', round(dose.get('guidance_applied_mean_abs_step_over_sigma', 0.), 6),
                  'nonzero', round(dose.get('guidance_applied_nonzero_fraction', 0.), 4),
                  flush=True)
        return report

    try:
        indices = np.random.default_rng(2026).permutation(len(np.load(data / 'valid_attrs_idx.npy')))
        screen = evaluate('screen', 'valid', indices[:8 if args.smoke else 256], 42, pool)
        ranked = rank([screen], pool)
        write_json(out / 'screen_ranking.json', ranked)
        selected = [r['config'] for r in ranked if r['eligible']][:3]
        if not args.smoke and selected:
            reports = [evaluate('refine', 'valid', indices[256:1024], s, selected)
                       for s in (1, 7, 42)]
            ranked = rank(reports, selected)
            write_json(out / 'refine_ranking.json', ranked)
        winner = [r['config'] for r in ranked if r['eligible']][:1] if selected else []
        write_json(out / 'frozen.json', winner)
        if not args.smoke and winner:
            for seed in (1, 7, 42):
                evaluate('confirm', 'valid', indices[1024:], seed, winner)
            test = np.arange(len(np.load(data / 'test_attrs_idx.npy')))
            for seed in (1, 7, 42):
                evaluate('test', 'test', test, seed, winner)
        write_json(out / 'status.json', dict(state='completed', smoke=args.smoke,
                                             screen_winners=len(selected)))
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
