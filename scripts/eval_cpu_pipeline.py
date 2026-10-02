"""Locked test evaluation of CPU CNN, pure VerbalTS and SAE latent MLP."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.steering import load_sae_checkpoint
from sae.shape_classifier import load_activation_cache, encode_sae_latents, load_classifier_checkpoint
from sae.provenance import write_json, sha256_file
from scripts.cpu_pipeline_metrics import accuracy_report, predict_cnn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    out, data = args.output_dir, args.data_root
    device = torch.device('cuda')
    labels = np.load(data / 'test_attrs_idx.npy')
    truth = np.load(data / 'test_ts.npy')
    cnn = PeakValleyClassifier1D(segment_len=43).to(device).eval()
    cnn.load_state_dict(torch.load(out / 'cnn/best.pt', map_location=device, weights_only=True))
    cnn_pred = predict_cnn(cnn, truth, device)
    report = dict(split='test', seeds=[1, 7, 42], n=len(labels),
        cnn=accuracy_report(cnn_pred, labels),
        always_nothing=accuracy_report(np.zeros_like(labels), labels),
        verbalts=[], mlp=[],
        caveats=['CNN-based VerbalTS score measures local morphology, not full caption correctness.',
                 'nothing means no injection, not absence of natural features.',
                 'MLP measures real-data noisy-forward latents, not generated-trajectory accuracy.'])
    model, _ = load_verbalts(args.checkpoint, device)
    embeddings = np.load(data / 'test_cap_emb.npy')
    with torch.no_grad():
        for seed in report['seeds']:
            curves = np.concatenate([generate_variant(model,
                torch.as_tensor(embeddings[start:start + 32], device=device).float(), None, seed + start)
                for start in range(0, len(labels), 32)])
            assert curves.shape == (len(labels), 128) and np.isfinite(curves).all()
            pred = predict_cnn(cnn, curves, device)
            report['verbalts'].append(dict(seed=seed, **accuracy_report(pred, labels)))
            np.savez(out / f'generated_seed{seed}.npz', curves=curves, predicted=pred, labels=labels)
            print('VerbalTS', report['verbalts'][-1], flush=True)
    del model
    torch.cuda.empty_cache()
    sae, t_range, _ = load_sae_checkpoint(out / 'sae/best.pt', device)
    assert tuple(t_range) == (5, 45)
    mlp, meta = load_classifier_checkpoint(out / 'mlp/best.pt', device)
    assert meta['sae_checkpoint_sha256'] == sha256_file(out / 'sae/best.pt')
    for seed in report['seeds']:
        samples, steps, target, payload = load_activation_cache(
            out / f'test_activations_{seed}/test_layer1_t5_45.pt',
            data / 'test_attrs_idx.npy', labels_from_attrs=True, num_heads=3)
        assert payload['source']['checkpoint_sha256'] == sha256_file(args.checkpoint)
        features = encode_sae_latents(samples, sae, device, 128)
        with torch.no_grad():
            pred = torch.cat([mlp(features[start:start + 128].to(device).float()).argmax(-1).cpu()
                              for start in range(0, len(features), 128)]).numpy()
        metrics = accuracy_report(pred, target.numpy())
        metrics['time_bins'] = {f'{lo}-{hi}': accuracy_report(pred[(steps >= lo) & (steps <= hi)],
            labels[(steps >= lo) & (steps <= hi)]) for lo, hi in ((5, 18), (19, 31), (32, 45))}
        report['mlp'].append(dict(seed=seed, **metrics))
        np.savez(out / f'mlp_seed{seed}.npz', predicted=pred, labels=labels, timesteps=steps.numpy())
    report['sae'] = json.loads((out / 'sae/metrics.json').read_text())['best_valid_recon_mse']
    report['checkpoint_hashes'] = {str(path): sha256_file(path) for path in
        (args.checkpoint, out / 'cnn/best.pt', out / 'sae/best.pt', out / 'mlp/best.pt')}
    lines = ['# CPU pipeline test report', '',
        'Accuracy = all three positions correct. VerbalTS is CNN-scored local morphology only.', '',
        '| Model | Whole-curve ACCR | Segment accuracy | Macro F1 |',
        '|---|---:|---:|---:|']
    for name in ('cnn', 'verbalts', 'mlp', 'always_nothing'):
        values = report[name] if isinstance(report[name], list) else [report[name]]
        cells = []
        for key in ('accr', 'segment_accuracy', 'macro_f1'):
            scores = np.array([m[key] for m in values]) * 100
            cells.append(f'{scores.mean():.2f}%' + (f' ± {scores.std():.2f} pp' if len(values) > 1 else ''))
        lines.append('| ' + ' | '.join([name] + cells) + ' |')
    lines += ['', 'Standard deviations are across fixed seeds, not confidence intervals.',
              f"SAE best validation reconstruction MSE: {report['sae']}", '', *report['caveats']]
    write_json(out / 'metrics.json', report)
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()