"""Interim three-model preview on a fixed validation subset, without tuning."""
import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from contsg.config.loader import load_config
from contsg.models.bridge import BridgeModule
from contsg.models.diffusets import DiffuSETS
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from scripts.cpu_pipeline_metrics import accuracy_report, predict_cnn
from scripts.train_three_model_candidate import exemplar_index

ROOT = Path(__file__).resolve().parents[1]


def snapshot_best(run, destination):
    paths = list(run.glob('*/checkpoints/finetune/**/loss=*.ckpt'))
    if not paths:
        raise FileNotFoundError(f'No finetune checkpoint in {run}')
    source = min(paths, key=lambda p: float(p.stem.split('=')[-1]))
    shutil.copyfile(source, destination)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--n-samples', type=int, default=256)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = ROOT / 'datasets/cpu_shape_v1'
    base = ROOT / 'results/cpu_shape_v1'
    attrs = np.load(data / 'valid_attrs_idx.npy')
    if not 1 <= args.n_samples <= len(attrs):
        raise ValueError('Invalid sample count')
    indices = np.random.default_rng(2026).permutation(len(attrs))[:args.n_samples]
    labels = attrs[indices]
    truth = np.load(data / 'valid_ts.npy')[indices, :, 0]
    emb = np.load(data / 'valid_cap_emb.npy')[indices].astype(np.float32)
    captions = np.load(data / 'valid_text_caps.npy', allow_pickle=True).reshape(-1)[indices]
    train = np.load(data / 'train_ts.npy', mmap_mode='r')
    exemplar_ids = np.array([exemplar_index(i, len(train)) for i in indices])
    device = torch.device(args.device)
    cnn = PeakValleyClassifier1D(segment_len=43).to(device).eval()
    cnn.load_state_dict(torch.load(base / 'pipeline/cnn/best.pt', map_location=device, weights_only=True))
    sources = {}
    for name, job in (('bridge', 1018523), ('diffusets', 1018524)):
        destination = out / f'{name}.ckpt'
        source = snapshot_best(base / f'{name}_{job}', destination)
        sources[name] = dict(source=str(source), sha256=sha256_file(destination))
    write_json(out / 'protocol.json', dict(split='valid', indices=indices.tolist(), seed=42,
        interim=True, sources=sources, bridge_exemplar_indices=exemplar_ids.tolist(),
        sampler='ddim', bridge_clip_denoised=False, batch_size=16))
    report, generated = {}, {}
    try:
        for name in ('verbalts', 'bridge', 'diffusets'):
            write_json(out / 'status.json', dict(state='running', model=name))
            if name == 'verbalts':
                ckpt = Path(json.loads((base / 'pipeline/inputs.json').read_text())['checkpoint'])
                model, _ = load_verbalts(ckpt, device)
                sources[name] = dict(source=str(ckpt), sha256=sha256_file(ckpt))
            else:
                config = load_config(config_path=ROOT / f'configs/generators/{name}_cpu_shape_v1.yaml')
                config.device = str(device)
                cls = BridgeModule if name == 'bridge' else DiffuSETS
                model = cls(config=config, use_condition=True)
                payload = torch.load(out / f'{name}.ckpt', map_location='cpu', weights_only=False)
                model.load_state_dict(payload['state_dict'], strict=True)
                sources[name]['epoch_zero_based'] = payload.get('epoch')
                del payload
                model = model.to(device).eval()
            parts = []
            for start in range(0, len(indices), 16):
                condition = torch.as_tensor(emb[start:start + 16], device=device)
                torch.manual_seed(42 + start)
                with torch.no_grad():
                    if name == 'verbalts':
                        curves = generate_variant(model, condition, None, 42 + start)
                    else:
                        kwargs = dict(sampler='ddim')
                        if name == 'bridge':
                            kwargs.update(clip_denoised=False, bridge_example_ts=torch.as_tensor(
                                np.array(train[exemplar_ids[start:start + 16]]), device=device).float())
                        curves = model.generate(condition, n_samples=1, **kwargs)[:, 0, :, 0].cpu().numpy()
                parts.append(curves)
            curves = np.concatenate(parts)
            assert curves.shape == truth.shape and np.isfinite(curves).all()
            predicted = predict_cnn(cnn, curves, device)
            report[name] = dict(**accuracy_report(predicted, labels),
                mse=float(np.mean((curves - truth) ** 2)), absolute_max=float(np.abs(curves).max()),
                checkpoint=sources[name])
            np.savez(out / f'{name}.npz', curves=curves, predicted=predicted, labels=labels, indices=indices)
            generated[name] = curves
            write_json(out / 'metrics.json', report)
            print(name, report[name], flush=True)
            del model
            torch.cuda.empty_cache()
        rng = np.random.default_rng(42)
        selected = []
        for cls in range(4):
            pool = np.flatnonzero(labels.max(1) == cls)
            selected.extend(rng.choice(pool, min(2, len(pool)), replace=False).tolist())
        fig, axes = plt.subplots(len(selected), 3, figsize=(18, 3 * len(selected)), squeeze=False)
        for row, i in enumerate(selected):
            for col, name in enumerate(generated):
                ax = axes[row, col]
                ax.plot(truth[i], label='Reference', linewidth=1.5)
                ax.plot(generated[name][i], label=name, linewidth=1.3)
                ax.set_title(f'{name} | valid #{indices[i]}\n' + textwrap.fill(str(captions[i]), 65), fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'comparison.png', dpi=140)
        plt.close(fig)
        write_json(out / 'plot_indices.json', indices[selected].tolist())
        lines = ['# Interim CPU baseline preview', '',
                 'Fixed validation subset, seed 42. No parameter selection. Training is still ongoing.',
                 'CNN scores local morphology only; reference MSE is not full caption compliance.', '',
                 '| Model | ACCR | Segment accuracy | MSE |', '|---|---:|---:|---:|']
        for name, m in report.items():
            lines.append(f"| {name} | {m['accr']:.2%} | {m['segment_accuracy']:.2%} | {m['mse']:.4f} |")
        lines += ['', '![Random cases, two per class](comparison.png)']
        (out / 'README.md').write_text('\n'.join(lines) + '\n')
        write_json(out / 'status.json', dict(state='completed'))
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()