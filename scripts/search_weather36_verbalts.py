"""Validation-only paired pilot; never reuse its parameters for other models."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import load_classifier_checkpoint
from sae.steering import LatentClassifierGuidanceWrapper, TimestepSAEWrapper, load_sae_checkpoint
from scripts.train_weather36_cnn import evaluate_curves

ROOT = Path(__file__).resolve().parents[1]


def pilot_indices(labels):
    rng = np.random.default_rng(2026)
    return np.concatenate([rng.permutation(np.flatnonzero((labels == c).all(1)))[:4]
                           for c in np.unique(labels, axis=0)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = ROOT / 'datasets/weather_three_stage_36_morph'
    latent = ROOT / 'results/weather36/verbalts_latents_1015328'
    paths = [ROOT / 'results/weather36/runs/20260926_204158_weather_three_stage_36_morph_verbalts_verbalts_lr0.001/checkpoints/finetune/finetune-epoch=64-val/loss=0.1318.ckpt',
             latent / 'models/t5_45/best.pt', latent / 'classwise_classifiers/t5_45/best.pt',
             ROOT / 'results/weather36/cnn/segment_cnn.pth']
    device = torch.device('cuda')
    model, _ = load_verbalts(paths[0], device)
    sae, window, _ = load_sae_checkpoint(paths[1], device)
    mlp, _ = load_classifier_checkpoint(paths[2], device)
    cnn = PeakValleyClassifier1D(segment_len=12).to(device)
    cnn.load_state_dict(torch.load(paths[3], map_location=device, weights_only=True))
    for net in (model, sae, mlp, cnn):
        net.eval().requires_grad_(False)
    labels = np.load(data / 'valid_attrs_idx.npy')
    indices = pilot_indices(labels)
    labels = labels[indices]
    truth = np.load(data / 'valid_ts.npy')[indices, :, 0]
    emb = np.load(data / 'valid_cap_emb.npy')[indices]
    configs = [dict(name='pure'), dict(name='sae')]
    configs += [dict(name=f'dynamic_eta{eta}', eta=eta, topk=32,
                     topk_mode='dynamic', adaptive_gamma=0., iters=1,
                     rel_cap=1., max_step=.5, batch_invariant=True)
                for eta in (1., 10., 100., 1000.)]
    report = dict(model='verbalts', split='valid', pilot_only=True,
                  indices=indices.tolist(), seeds=[1, 7, 42], batch_size=12,
                  checkpoints={str(p): sha256_file(p) for p in paths},
                  data_hashes={n: sha256_file(data / n) for n in
                               ('valid_ts.npy', 'valid_attrs_idx.npy', 'valid_cap_emb.npy')},
                  cttp_status='not_scored', results=[], ranking=[])
    write_json(args.output_dir / 'summary.json', report)
    for config in configs:
        wrapper = None
        if config['name'] == 'sae':
            wrapper = TimestepSAEWrapper(sae, window)
        elif config['name'] != 'pure':
            wrapper = LatentClassifierGuidanceWrapper(
                sae, window, mlp, **{k: v for k, v in config.items() if k != 'name'})
        if wrapper is not None:
            wrapper.to(device).eval()
        for seed in report['seeds']:
            parts = []
            for start in range(0, len(indices), 12):
                if isinstance(wrapper, LatentClassifierGuidanceWrapper):
                    wrapper.set_targets(torch.as_tensor(labels[start:start+12], device=device).long())
                with torch.no_grad():
                    parts.append(generate_variant(model, torch.as_tensor(
                        emb[start:start+12], device=device).float(), wrapper, seed + start))
            curves = np.concatenate(parts)
            if curves.shape != truth.shape or not np.isfinite(curves).all():
                raise ValueError('Invalid generated shape or non-finite values')
            x = torch.from_numpy(curves.reshape(-1, 1, 12))
            dummy = torch.zeros(len(x), dtype=torch.long)
            metrics = evaluate_curves(cnn, TensorDataset(x, dummy, dummy), labels, device, 256)
            mse = ((curves - truth) ** 2).mean(1)
            np.savez(args.output_dir / f'{config["name"]}_seed{seed}.npz',
                     curves=curves, indices=indices, labels=labels, mse=mse)
            row = dict(config=config, seed=seed, cnn=metrics, mse=float(mse.mean()),
                       audit={} if wrapper is None else wrapper.audit())
            report['results'].append(row)
            write_json(args.output_dir / 'summary.json', report)
            print(config['name'], seed, metrics['accr'], row['mse'], flush=True)
    for config in configs:
        rows = [r for r in report['results'] if r['config'] == config]
        scores = [r['cnn']['accr'] for r in rows]
        report['ranking'].append(dict(config=config, accr_mean=float(np.mean(scores)),
                                      accr_std=float(np.std(scores)),
                                      mse_mean=float(np.mean([r['mse'] for r in rows]))))
    report['ranking'].sort(key=lambda r: (-r['accr_mean'], r['mse_mean']))
    write_json(args.output_dir / 'summary.json', report)
    model.verbalts.activation_transform = None


if __name__ == '__main__':
    main()