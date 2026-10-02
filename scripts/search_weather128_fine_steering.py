"""Bounded four-stage steering pilot for the 5-class fine labels.

Targets are the raw fine labels (1..5); the latent classifier has 6 classes
per head (0 = nothing, 1..5 = fine shapes), and the guidance wrapper infers
the class count from the classifier output layer.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import load_classifier_checkpoint
from sae.steering import LatentClassifierGuidanceWrapper, TimestepSAEWrapper, load_sae_checkpoint
from scripts.fine_weather128_models import FinePeakValleyClassifier1D, evaluate_fine_curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cnn', 'sae', 'mlp', 'data-root', 'search-indices',
                 'confirm-indices', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--pilot-samples-per-background', type=int, default=80)
    args = p.parse_args()
    if args.pilot_samples_per_background < 1:
        p.error('pilot-samples-per-background must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    model, _ = load_verbalts(args.checkpoint, device)
    sae, window, _ = load_sae_checkpoint(args.sae, device)
    mlp, _ = load_classifier_checkpoint(args.mlp, device)
    cnn = FinePeakValleyClassifier1D(segment_len=32).to(device)
    cnn.load_state_dict(torch.load(args.cnn, map_location=device, weights_only=True))
    for net in (model, sae, mlp, cnn):
        net.eval().requires_grad_(False)
    labels = np.load(args.data_root / 'valid_attrs_idx.npy')
    emb = np.load(args.data_root / 'valid_cap_emb.npy')
    truth = np.load(args.data_root / 'valid_ts.npy')[:, :, 0]
    ids = np.load(args.data_root / 'valid_background_ids.npy')
    search, confirm = np.load(args.search_indices), np.load(args.confirm_indices)
    assert not set(ids[search]) & set(ids[confirm])
    # Four full backgrounds with a bounded sample budget per background.
    rng = np.random.default_rng(20260928)
    pilot = []
    for background in rng.permutation(np.unique(ids[search]))[:4]:
        rows = np.flatnonzero(ids[search] == background)
        pick = rng.choice(rows, size=min(args.pilot_samples_per_background, len(rows)),
                          replace=False)
        pilot.append(search[pick])
    pilot = np.sort(np.concatenate(pilot))
    configs = [dict(name='pure'), dict(name='sae_only')]
    configs += [dict(name=f'dynamic_eta{eta:g}', eta=eta, topk=32,
                    topk_mode='dynamic', adaptive_gamma=0., iters=1,
                    rel_cap=1., max_step=.5, batch_invariant=True)
                for eta in (1., 10., 100., 1000.)]
    report = dict(status='RUNNING', split='valid', test_evaluated=False,
                  note='Confirmation backgrounds excluded from steering search, not from model validation.',
                  checkpoints={str(v): sha256_file(v) for v in
                               (args.checkpoint, args.cnn, args.sae, args.mlp)}, results=[])

    def run(config, indices, seed, phase):
        wrapper = None
        if config['name'] == 'sae_only':
            wrapper = TimestepSAEWrapper(sae, window)
        elif config['name'].startswith('dynamic'):
            wrapper = LatentClassifierGuidanceWrapper(
                sae, window, mlp, **{k: v for k, v in config.items() if k != 'name'})
        if wrapper is not None:
            wrapper.to(device).eval()
        parts = []
        for start in range(0, len(indices), 32):
            batch = indices[start:start + 32]
            if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                wrapper.set_targets(torch.as_tensor(labels[batch], device=device).long())
            with torch.no_grad():
                parts.append(generate_variant(model, torch.as_tensor(emb[batch], device=device).float(),
                                              wrapper, seed + start))
        curves = np.concatenate(parts)
        assert curves.shape == truth[indices].shape and np.isfinite(curves).all()
        dummy = torch.zeros(len(curves) * 4, dtype=torch.long)
        metrics = evaluate_fine_curves(cnn, TensorDataset(torch.from_numpy(curves.reshape(-1, 1, 32)),
                                                          dummy, dummy), labels[indices], device)
        correct = np.asarray(metrics.pop('correct'))
        name = f'{phase}_{config["name"]}_seed{seed}'
        np.savez(args.output_dir / f'{name}.npz', curves=curves, indices=indices,
                 labels=labels[indices], correct=correct)
        row = dict(phase=phase, config=config, seed=seed, cnn=metrics,
                   paired_mse=float(((curves - truth[indices]) ** 2).mean()),
                   audit={} if wrapper is None else wrapper.audit())
        report['results'].append(row)
        write_json(args.output_dir / 'summary.json', report)
        print(name, metrics, flush=True)
        return metrics['accr']

    scores = {c['name']: np.mean([run(c, pilot, s, 'pilot') for s in (1, 7, 42)]) for c in configs}
    winner = max(configs[2:], key=lambda c: scores[c['name']])
    report['pilot_scores'] = scores
    report['selected_dynamic'] = winner
    for config in (configs[0], configs[1], winner):
        for seed in (1, 7, 42):
            run(config, confirm, seed, 'confirm')
    report['status'] = 'COMPLETE_REVIEW_CURVE_QUALITY'
    write_json(args.output_dir / 'summary.json', report)
    model.verbalts.activation_transform = None


if __name__ == '__main__':
    main()
