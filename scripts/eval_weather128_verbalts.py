"""Validation-only pure VerbalTS joint four-stage ACCR."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from scripts.train_weather128_cnn import evaluate_curves, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cnn', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--indices', type=Path, help='Validation sample indices only')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    model, state_hash = load_verbalts(args.checkpoint, device)
    assert model.config.data.seq_length == 128
    assert model.config.data.caption_variant == 'base'
    model.eval().requires_grad_(False)
    cnn = PeakValleyClassifier1D(segment_len=32).to(device).eval()
    cnn.load_state_dict(torch.load(args.cnn, map_location=device, weights_only=True))
    dataset, labels = load_split(args.data_root, 'valid')
    truth = np.load(args.data_root / 'valid_ts.npy')[:, :, 0]
    emb = np.load(args.data_root / 'valid_cap_emb.npy')
    assert emb.shape == (len(labels), 1024)
    indices = np.arange(len(labels))
    if args.indices:
        indices = np.load(args.indices)
        assert indices.ndim == 1 and len(indices) and np.issubdtype(indices.dtype, np.integer)
        assert len(np.unique(indices)) == len(indices) and indices.min() >= 0 and indices.max() < len(labels)
        truth, labels, emb = truth[indices], labels[indices], emb[indices]
        dummy = torch.zeros(len(labels) * 4, dtype=torch.long)
        dataset = TensorDataset(torch.from_numpy(truth.reshape(-1, 1, 32)), dummy, dummy)
    control = evaluate_curves(cnn, dataset, labels, device)
    control.pop('correct')
    report = dict(split='valid', variant='pure', caption_variant='base',
                  seed=args.seed, batch_size=args.batch_size, sampler='ddim',
                  samples_per_caption=1, test_evaluated=False,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
                  state_sha256=state_hash, cnn_sha256=sha256_file(args.cnn),
                  indices=indices.tolist(),
                  real_data_control=control, status='GENERATING')
    write_json(args.output_dir / 'report.json', report)
    parts = []
    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            condition = torch.as_tensor(emb[start:start + args.batch_size], device=device).float()
            parts.append(generate_variant(model, condition, None, args.seed + start))
            print(f'Generated {min(start + args.batch_size, len(labels))}/{len(labels)}', flush=True)
    curves = np.concatenate(parts)
    assert curves.shape == truth.shape and np.isfinite(curves).all()
    x = torch.from_numpy(curves.reshape(-1, 1, 32))
    dummy = torch.zeros(len(x), dtype=torch.long)
    metrics = evaluate_curves(cnn, TensorDataset(x, dummy, dummy), labels, device)
    correct = np.asarray(metrics.pop('correct'), dtype=bool)
    metrics['all_correct_count'] = int(correct.all(axis=1).sum())
    mse = ((curves - truth) ** 2).mean(axis=1)
    np.savez(args.output_dir / 'samples.npz', curves=curves, labels=labels,
             correct=correct, paired_mse=mse)
    report.update(status='COMPLETE', cnn=metrics, paired_mse=float(mse.mean()))
    write_json(args.output_dir / 'report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()