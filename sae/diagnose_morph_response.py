"""Pulse/window steering on captured SAE-only DDIM states (validation only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from contsg.eval.metrics.segment import PeakValleyClassifier1D
from sae.activations import load_verbalts
from sae.eval_guidance import generate_variant
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import load_classifier_checkpoint
from sae.shapes import SHAPE_NAMES, SHAPE_TO_TARGET, classify_curves
from sae.steering import LatentClassifierGuidanceWrapper, TimestepSAEWrapper, load_sae_checkpoint


class CaptureTransform(nn.Module):
    """Observe hook input/output without modifying the wrapped computation."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.before = self.after = None

    def forward(self, hidden, step):
        output, latent = self.wrapped(hidden, step)
        self.before = hidden.detach().clone()
        self.after = output.detach().clone()
        return output, latent


def rmse_rows(a, b):
    return (a - b).float().reshape(a.shape[0], -1).square().mean(1).sqrt()


def guidance_steps(start, length, window):
    if length < 1 or start > window[1] or start - length + 1 < window[0]:
        raise ValueError('entire intervention must lie within SAE window')
    return list(range(start, start - length, -1))


def target_probabilities(predictions, target_names):
    """Two-head target probability product, matching compute_cnn_report."""
    values = np.empty(target_names.shape, dtype=np.float64)
    for i, row in enumerate(predictions):
        for stage, prediction in enumerate(row):
            peak, valley = SHAPE_TO_TARGET[str(target_names[i, stage])]
            values[i, stage] = (prediction['peak_probabilities'][peak]
                                * prediction['valley_probabilities'][valley])
    return values


def probability_metrics(predictions, base_predictions, target_names):
    baseline = target_probabilities(base_predictions, target_names)
    values = target_probabilities(predictions, target_names)
    delta = values - baseline
    incorrect = np.array([[p['shape'] for p in row] for row in base_predictions]) != target_names
    return dict(target_probability=values.tolist(), target_probability_delta=delta.tolist(),
                mean_delta_per_stage=delta.mean(0).tolist(),
                improved_per_stage=(delta > 1e-6).sum(0).tolist(),
                worsened_per_stage=(delta < -1e-6).sum(0).tolist(),
                baseline_incorrect_count_per_stage=incorrect.sum(0).tolist(),
                mean_delta_baseline_incorrect_per_stage=[
                    float(delta[incorrect[:, j], j].mean()) if incorrect[:, j].any() else None
                    for j in range(3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, nargs='+', default=[40, 25, 10])
    parser.add_argument('--intervention-length', type=int, default=1)
    args = parser.parse_args()
    source = json.loads((args.run / 'summary.json').read_text())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    for path, digest in source['checkpoints'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f'checkpoint changed: {path}')
    for name, digest in source['data_sha256'].items():
        if sha256_file(args.data_root / name) != digest:
            raise ValueError(f'data changed: {name}')
    paths = list(source['checkpoints'])
    model_path = next(p for p in paths if p.endswith('.ckpt'))
    sae_path = next(p for p in paths if '/models/' in p)
    mlp_path = next(p for p in paths if '/classwise_classifiers/' in p)
    cnn_path = next(p for p in paths if p.endswith('segment_cnn.pth'))
    device = torch.device(args.device)
    model, _ = load_verbalts(model_path, device)
    sae, window, _ = load_sae_checkpoint(sae_path, device)
    mlp, _ = load_classifier_checkpoint(mlp_path, device)
    cnn = PeakValleyClassifier1D(segment_len=43).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    for net in (model, sae, mlp, cnn):
        net.eval().requires_grad_(False)
    if any(t < window[0] or t > window[1] or t >= model.num_steps for t in args.steps):
        raise ValueError('pulse steps must be in SAE and diffusion windows')
    schedules = {t: guidance_steps(t, args.intervention_length, window) for t in args.steps}
    indices = source['indices'][:source['batch_size']]
    batch = len(indices)
    targets = np.load(args.data_root / 'valid_attrs_idx.npy')[indices]
    condition = torch.tensor(np.load(args.data_root / 'valid_cap_emb.npy')[indices],
                             dtype=torch.float32, device=device)
    baseline = TimestepSAEWrapper(sae, window).to(device).eval()
    states = {}

    def capture(module, inputs):
        t = int(inputs[3][0])
        states[t] = dict(inputs=tuple(v.detach().clone() if torch.is_tensor(v) else v for v in inputs),
                         cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

    handle = model.verbalts.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            curves = generate_variant(model, condition, baseline, source['seed'])
    finally:
        handle.remove()
    previous = np.load(args.run / 'sae.npy')[:batch]
    np.testing.assert_allclose(curves, previous, rtol=0, atol=1e-6)
    np.save(out / 'sae.npy', curves)
    base_predictions = classify_curves(cnn, curves, device)
    target_names = np.asarray(SHAPE_NAMES)[targets]
    base_correct = np.array([[p['shape'] for p in row] for row in base_predictions]) == target_names
    report = dict(indices=indices, seed=source['seed'], batch_size=batch,
                  steps=args.steps, intervention_length=args.intervention_length,
                  schedules=schedules, sampler='ddim_eta0', continuation='sae_only_after_window',
                  checkpoints=source['checkpoints'], data_sha256=source['data_sha256'],
                  source_sha256={str(p): sha256_file(p) for p in
                                 [Path(__file__).resolve(), Path(__file__).with_name('steering.py'),
                                  Path(__file__).parents[1] / 'contsg/models/verbalts.py']},
                  old_source_matches={p: sha256_file(Path(p)) == h for p, h in source['source_sha256'].items()},
                  baseline_max_abs_vs_previous=float(np.abs(curves - previous).max()), results={})
    with torch.no_grad():
        for pulse in args.steps:
            reference_hidden = reference_noise = reference_x0 = reference_next = None
            for name in ['sae', 'reference', 'active_only']:
                wrapper = baseline if name == 'sae' else LatentClassifierGuidanceWrapper(
                    sae, window, mlp, eta=5120, adaptive_gamma=6, topk=32,
                    topk_mode='dynamic', full_strength_shapes='', batch_invariant=False,
                    active_only=name == 'active_only', diagnostic_trace=True)
                if name != 'sae':
                    wrapper.set_targets(torch.tensor(targets, dtype=torch.long, device=device))
                observer = CaptureTransform(wrapper).to(device).eval()
                saved = states[pulse]
                torch.set_rng_state(saved['cpu_rng'])
                if device.type == 'cuda':
                    torch.cuda.set_rng_state(saved['cuda_rng'], device)
                x = saved['inputs'][0].clone()
                for t_int in range(pulse, -1, -1):
                    _, tp, attr, t = states[t_int]['inputs']
                    model.verbalts.activation_transform = observer if t_int in schedules[pulse] else baseline
                    noise = model.verbalts(x, tp, attr, t)[0].unsqueeze(1)
                    if t_int == pulse:
                        pulse_input = observer.before.clone()
                        hidden = observer.after.reshape(batch, -1, observer.after.shape[-1])
                        x0 = ((x - model.one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1) * noise)
                              * model.alpha_bar_sqrt_inverse[t].view(-1, 1, 1, 1))
                        pulse_noise = noise.clone()
                    x = model._ddim_reverse(x, noise, t)
                    if t_int == pulse:
                        next_x = x.clone()
                final = x[:, 0, 0].cpu().numpy()
                if not np.isfinite(final).all():
                    raise ValueError('non-finite continuation')
                if name == 'sae':
                    np.testing.assert_allclose(final, curves, rtol=0, atol=1e-6)
                    reference_hidden, reference_noise = hidden.clone(), pulse_noise.clone()
                    reference_x0, reference_next = x0.clone(), next_x.clone()
                    reference_input = pulse_input.clone()
                if not torch.equal(pulse_input, reference_input):
                    raise AssertionError('pulse hidden inputs differ')
                predictions = classify_curves(cnn, final, device)
                correct = np.array([[p['shape'] for p in row] for row in predictions]) == target_names
                key = f't{pulse}_{name}'
                np.savez_compressed(out / f'{key}.npz', final=final, x_t=saved['inputs'][0].cpu().numpy(),
                                    noise=pulse_noise.cpu().numpy(), x0=x0.cpu().numpy(),
                                    next_x=next_x.cpu().numpy())
                metrics = dict(hidden_rmse=rmse_rows(hidden, reference_hidden).tolist(),
                               noise_rmse=rmse_rows(pulse_noise, reference_noise).tolist(),
                               x0_rmse=rmse_rows(x0, reference_x0).tolist(),
                               next_x_rmse=rmse_rows(next_x, reference_next).tolist(),
                               final_rmse=np.sqrt(((final - curves) ** 2).mean(1)).tolist(),
                               fixes_per_stage=(correct & ~base_correct).sum(0).tolist(),
                               breaks_per_stage=(~correct & base_correct).sum(0).tolist(),
                               predictions=predictions)
                metrics['probability'] = probability_metrics(predictions, base_predictions, target_names)
                metrics['applied_guidance_steps'] = [] if name == 'sae' else schedules[pulse]
                report['results'][key] = metrics
                if name != 'sae':
                    write_json(out / f'{key}_trace.json', wrapper.trace_records)
                write_json(out / 'summary.json', report)
                print(key, {k: v for k, v in metrics.items() if k != 'predictions'}, flush=True)
    model.verbalts.activation_transform = None
    print('RESPONSE_DIAGNOSTIC_COMPLETED', flush=True)


if __name__ == '__main__':
    main()