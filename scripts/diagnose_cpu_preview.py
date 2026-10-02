"""Audit a frozen CPU preview and separate VAE reconstruction from sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contsg.config.loader import load_config
from contsg.models.diffusets import DiffuSETS
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.provenance import sha256_file, write_json
from scripts.cpu_pipeline_metrics import accuracy_report, predict_cnn

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    preview = args.preview_dir
    protocol = json.loads((preview / 'protocol.json').read_text())
    metrics = json.loads((preview / 'metrics.json').read_text())
    data = ROOT / 'datasets/cpu_shape_v1'
    indices = np.array(protocol['indices'])
    truth = np.load(data / 'valid_ts.npy')[indices, :, 0]
    labels = np.load(data / 'valid_attrs_idx.npy')[indices]
    report = {'split': 'valid', 'n': len(indices), 'models': {}}
    for name in ('verbalts', 'bridge', 'diffusets'):
        saved = np.load(preview / f'{name}.npz')
        np.testing.assert_array_equal(saved['indices'], indices)
        np.testing.assert_array_equal(saved['labels'], labels)
        curves = saved['curves']
        assert np.isfinite(curves).all()
        np.testing.assert_allclose(np.mean((curves - truth) ** 2), metrics[name]['mse'])
        report['models'][name] = {
            'mse_by_segment': [float(np.mean((a - b) ** 2)) for a, b in
                               zip(np.array_split(curves, 3, axis=1), np.array_split(truth, 3, axis=1))],
            'absolute_max': float(np.abs(curves).max()),
            'curve_absolute_max_quantiles': np.quantile(np.abs(curves).max(1), [.5, .95, 1]).tolist(),
        }
        if name != 'verbalts':
            digest = sha256_file(preview / f'{name}.ckpt')
            assert digest == metrics[name]['checkpoint']['sha256']
            report['models'][name]['verified_sha256'] = digest
    config = load_config(config_path=ROOT / 'configs/generators/diffusets_cpu_shape_v1.yaml')
    config.device = 'cpu'
    model = DiffuSETS(config).eval()
    payload = torch.load(preview / 'diffusets.ckpt', map_location='cpu', weights_only=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    cnn = PeakValleyClassifier1D(segment_len=43).eval()
    cnn.load_state_dict(torch.load(ROOT / 'results/cpu_shape_v1/pipeline/cnn/best.pt',
                                   map_location='cpu', weights_only=True))
    sampled, means = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            x = torch.as_tensor(truth[start:start + 16, :, None]).float()
            z, mean, _ = model.encode(x)
            sampled.append(model.decode(z)[:, :truth.shape[1], 0].numpy())
            means.append(model.decode(mean * 0.18215)[:, :truth.shape[1], 0].numpy())
    arrays = {}
    for name, parts in (('vae_sampled', sampled), ('vae_mean', means)):
        curves = np.concatenate(parts)
        assert curves.shape == truth.shape and np.isfinite(curves).all()
        predicted = predict_cnn(cnn, curves, torch.device('cpu'))
        report[name] = dict(**accuracy_report(predicted, labels),
                           mse=float(np.mean((curves - truth) ** 2)),
                           absolute_max=float(np.abs(curves).max()))
        arrays[name] = curves
    np.savez(args.output_dir / 'reconstructions.npz', **arrays, indices=indices, labels=labels)
    write_json(args.output_dir / 'diagnostics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()