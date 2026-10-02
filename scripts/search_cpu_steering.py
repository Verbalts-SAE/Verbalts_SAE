"""Validation-only dynamic steering selection, followed by frozen test evaluation."""
import argparse
import itertools
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


def candidates():
    return [dict(name=f'dynamic_{i:02d}', topk=k, eta=e, gamma=g,
                 guidance_t_range=list(w))
            for i, (k, e, g, w) in enumerate(itertools.product(
                (16, 48), (320, 1280, 5120), (0, 3), ((5, 45), (5, 25))))]


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
    p.add_argument('--smoke', action='store_true')
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
    write_json(out / 'protocol.json', dict(candidates=candidates(), batch_size=16,
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
                    topk_mode='dynamic', guidance_t_range=config['guidance_t_range'],
                    rel_cap=1., max_step=.5, selection_score='applied', preserve_residual=True)
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
            report[name] = dict(**accuracy_report(predicted, labels),
                mse=float(np.mean((curves - truth) ** 2)),
                roughness=float(np.abs(np.diff(curves)).mean()),
                curvature=float(np.abs(np.diff(curves, n=2)).mean()),
                amplitude_p99=float(np.quantile(np.abs(curves).max(1), .99)))
            write_json(folder / 'metrics.json', report)
            print(stage, seed, name, report[name]['accr'], flush=True)
        return report

    try:
        indices = np.random.default_rng(2026).permutation(len(np.load(data / 'valid_attrs_idx.npy')))
        pool = candidates()[:1] if args.smoke else candidates()
        screen = evaluate('screen', 'valid', indices[:4 if args.smoke else 256], 42, pool)
        ranked = rank([screen], pool)
        write_json(out / 'screen_ranking.json', ranked)
        if not args.smoke:
            selected = [r['config'] for r in ranked if r['eligible']][:3]
            reports = [evaluate('refine', 'valid', indices[256:1024], s, selected) for s in (1, 7, 42)]
            ranked = rank(reports, selected)
            write_json(out / 'refine_ranking.json', ranked)
            winner = [r['config'] for r in ranked if r['eligible']][:1]
            write_json(out / 'frozen.json', winner)
            for seed in (1, 7, 42):
                evaluate('confirm', 'valid', indices[1024:], seed, winner)
            if winner:
                test = np.arange(len(np.load(data / 'test_attrs_idx.npy')))
                for seed in (1, 7, 42):
                    evaluate('test', 'test', test, seed, winner)
        write_json(out / 'status.json', dict(state='completed', smoke=args.smoke))
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()