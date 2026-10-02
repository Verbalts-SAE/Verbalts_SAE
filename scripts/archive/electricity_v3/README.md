# Archived electricity v3 validation jobs

This directory contains one-off SLURM entry points used to diagnose the first
electricity v3 SAE steering pipeline. They are retained for provenance, but
they are **not** the entry points for the next experiment iteration.

Archived jobs:

- `eval_electricity_v3_steering.slurm`: 128-sample steering sweep.
- `diagnose_electricity_v3_steering.slurm`: steering variants plus traces and
  diagnostic figures.
- `diagnose_electricity_v3_response.slurm`: single-step response propagation
  based on the old `diagnose128_1008498` output.
- `diagnose_electricity_v3_window.slurm`: length-1/length-5 intervention-window
  comparison based on the same old output.
- `diagnose_electricity_v3_pure.slurm`: pure VerbalTS caption-conditioning
  diagnostic.

The corresponding generated artifacts and logs were removed during cleanup:

- `results/electricity_v3/steering/`
- `results/electricity_v3/pure_diagnostics/`
- old files in `results/electricity_v3/logs/`

These scripts contain hard-coded paths and old run IDs. Do not submit them
without deliberately updating their dataset, checkpoint, output directory,
target-swap protocol, and evaluation design.

Still-active v3 reproducibility entry points remain in `scripts/`:

- `train_electricity_v3.slurm`
- `train_electricity_v3_downstream.slurm`
