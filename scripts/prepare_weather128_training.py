"""Independent configs; no test evaluation and no legacy scoring checkpoints."""
import json
from pathlib import Path
import yaml
from contsg.config.schema import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/weather128'
DATA = ROOT / 'datasets/weather_four_stage_128_morph_v2'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = OUT / 'configs'
    configs.mkdir(exist_ok=False)
    for lr in (1e-4, 3e-4, 1e-3):
        cfg = yaml.safe_load((ROOT / 'configs/generators/verbalts_electricity_v3.yaml').read_text())
        cfg['data'].update(name='weather_four_stage_128_morph', data_folder=str(DATA))
        # num_stages belongs to the diffusion-step projector, not morphology labels.
        cfg['eval'].update(cache_folder=str(OUT / 'cache'), use_cache=False)
        cfg['eval']['segment_classifier'].update(segment_len=32, n_segments=4, enable=False)
        cfg['train'].update(epochs=120, lr=lr, batch_size=32, accumulate_grad_batches=8,
                            log_grad_norm=False, log_param_norm=False)
        cfg['train']['stages'][0].update(epochs=120, lr=lr, early_stopping_patience=20)
        ExperimentConfig(**cfg)
        (configs / f'verbalts_lr{lr:g}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfg = yaml.safe_load((ROOT / 'configs/cttp/cttp_weather36.yaml').read_text())
    cfg['data'].update(name='weather_four_stage_128_morph', data_folder=str(DATA), seq_length=128)
    ExperimentConfig(**cfg)
    (configs / 'cttp.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / 'protocol.json').write_text(json.dumps(dict(
        status='BOOTSTRAP_ONLY', data=str(DATA), selection_split='valid', test_evaluated=False,
        seeds=[1,7,42], variants=['pure','sae_only','dynamic'],
        primary='four_stage_joint_ACCR', secondary=['stage_accuracy','CTTP','paired_MSE'],
        stage_bounds=[[0,32],[32,64],[64,96],[96,128]],
        required_before_final=['CNN quality review', 'VerbalTS validation selection',
            'new SAE training', 'four-head MLP training', 'steering gradient/perturbation smoke',
            'CTTP validation', 'validation search and independent background confirmation',
            'freeze all parameters before test']), indent=2))


if __name__ == '__main__':
    prepare()