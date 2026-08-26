"""DiffLens-style multiplicative latent steering wrapper.

Intervention: inside the SAE diffusion-timestep window, every in-window
sample whose caption targets a located class receives a FIXED multiplicative
edit on the located latent dimensions::

    z_new[d] = z[d] * boost_factor     for the top-k positive-IG dims
    z_new[d] = z[d] * suppress_factor  for the top-k negative-IG dims
    z_new[d] = z[d]                    otherwise

The edited latents are decoded through the frozen SAE decoder, so the edit
enters the residual stream through the same reconstruction path as the
classifier-guidance pipeline.  A pure multiplication leaves inactive
(``z == 0``) boost dims untouched — exactly the DiffLens "multiply_all"
recipe.  Samples whose caption does not target any located class keep the
unmodified SAE reconstruction.

Multi-class conflicts: when one sample's caption targets several edited
groups (e.g. "Beginning: single peak" and "End: double peaks"), the group
ratio vectors are merged by per-dimension multiplication, then clamped to
``[min_ratio, max_ratio]``.  Opposing roles cancel exactly
(2.0 * 0.5 == 1.0) while same-direction edits never stack beyond a single
group's strength by default.  Merged samples are counted in
``audit()["multi_hit_samples"]``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch

from contsg.models.sae_module import TopKSparseAutoencoder
from sae.shapes import SHAPE_NAMES, STAGE_NAMES
from sae.steering import TimestepSAEWrapper

_STAGE_LOOKUP = {name.lower(): index for index, name in enumerate(STAGE_NAMES)}
_SHAPE_LOOKUP = {name: index for index, name in enumerate(SHAPE_NAMES)}


def parse_group_key(key: str) -> tuple[int, int]:
    """Parse ``"beginning/single peak"`` into ``(stage_index, shape_index)``."""

    parts = str(key).strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"group key must be 'stage/shape', got {key!r}")
    stage_name = parts[0].strip().lower()
    shape_name = parts[1].strip()
    if stage_name not in _STAGE_LOOKUP:
        raise ValueError(f"unknown stage name {parts[0]!r} in group key {key!r}")
    if shape_name not in _SHAPE_LOOKUP:
        raise ValueError(f"unknown shape name {parts[1]!r} in group key {key!r}")
    return _STAGE_LOOKUP[stage_name], _SHAPE_LOOKUP[shape_name]


class MultiplicativeSteeringWrapper(TimestepSAEWrapper):
    """Fixed per-class multiplicative edit on SAE latents inside the t-window.

    ``edits`` maps a group key (``"stage/shape"``) to a dict with the two
    dimension lists ``{"boost": [...], "suppress": [...]}`` located by
    :mod:`Baseline.ig_attribution`.  Every group gets its own ratio vector
    (ones everywhere except the edited dims); at forward time a sample uses
    the ratio vector of the group its caption targets.  A sample targeting
    several edited groups merges the group ratios by per-dimension
    multiplication and then clamps to ``[min_ratio, max_ratio]``: opposing
    roles on a dim cancel exactly (2.0 * 0.5 == 1.0) while same-direction
    edits never stack beyond a single group's strength by default.
    """

    def __init__(
        self,
        sae: TopKSparseAutoencoder,
        t_range: Tuple[int, int],
        edits: Mapping[str, Mapping[str, Sequence[int]]],
        boost_factor: float = 2.0,
        suppress_factor: float = 0.5,
        max_ratio: float | None = None,
        min_ratio: float | None = None,
    ) -> None:
        # Set audit counters before super().__init__: TimestepSAEWrapper's
        # constructor calls self.reset_audit(), which the override below
        # extends with the counters declared here.
        self.steered_samples = 0
        self.steered_rows = 0
        self.delta_sum = 0.0
        self.multi_hit_samples = 0
        super().__init__(sae, t_range)

        self.boost_factor = float(boost_factor)
        self.suppress_factor = float(suppress_factor)
        if self.boost_factor <= 0.0:
            raise ValueError("boost_factor must be positive")
        if self.suppress_factor < 0.0:
            raise ValueError("suppress_factor must be non-negative")
        # Default clamp keeps merged ratios within a single group's strength:
        # same-direction edits never stack beyond one edit, opposing edits
        # cancel to 1.0.  Pass explicit values to widen or disable (None).
        self.max_ratio = (
            max(1.0, self.boost_factor) if max_ratio is None else float(max_ratio)
        )
        self.min_ratio = (
            min(1.0, self.suppress_factor) if min_ratio is None else float(min_ratio)
        )
        if self.min_ratio > self.max_ratio:
            raise ValueError(
                f"min_ratio {self.min_ratio} must not exceed max_ratio {self.max_ratio}"
            )

        self.group_ratios: dict[tuple[int, int], torch.Tensor] = {}
        for index, (key, spec) in enumerate(dict(edits).items()):
            stage, shape = parse_group_key(key)
            boost = [int(dim) for dim in spec.get("boost", [])]
            suppress = [int(dim) for dim in spec.get("suppress", [])]
            dims = boost + suppress
            if not dims:
                raise ValueError(f"edit group {key!r} lists no dimensions")
            if any(dim < 0 or dim >= int(sae.latent_dim) for dim in dims):
                raise ValueError(
                    f"edit group {key!r} has dims outside [0, {sae.latent_dim})"
                )
            if set(boost) & set(suppress):
                raise ValueError(f"edit group {key!r} boost/suppress dims overlap")
            ratio = torch.ones(int(sae.latent_dim), dtype=torch.float32)
            if boost:
                ratio[boost] = self.boost_factor
            if suppress:
                ratio[suppress] = self.suppress_factor
            self.register_buffer(f"group_ratio_{index}", ratio)
            self.group_ratios[(stage, shape)] = ratio

        initial = torch.empty((0, 3), dtype=torch.long)
        self.register_buffer("target_classes", initial)

    def reset_audit(self) -> None:
        super().reset_audit()
        self.steered_samples = 0
        self.steered_rows = 0
        self.delta_sum = 0.0
        self.multi_hit_samples = 0

    def set_targets(self, target_classes: torch.Tensor) -> None:
        """Set caption-derived targets shaped ``(batch, 3)`` (shape id per stage)."""

        values = torch.as_tensor(target_classes, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != len(STAGE_NAMES):
            raise ValueError(
                "target_classes must have shape (batch, 3), got "
                f"{tuple(values.shape)}"
            )
        if values.min() < 0 or values.max() >= len(SHAPE_NAMES):
            raise ValueError(
                f"target class indices must be in [0, {len(SHAPE_NAMES) - 1}]"
            )
        self.target_classes = values.detach().clone()

    def audit(self) -> dict[str, float | int]:
        report = super().audit()
        report["steered_samples"] = self.steered_samples
        report["steered_rows"] = self.steered_rows
        report["mean_abs_delta"] = self.delta_sum / max(self.steered_rows, 1)
        report["multi_hit_samples"] = self.multi_hit_samples
        return report

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed, latents = super().forward(hidden, diffusion_step)
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        if self.target_classes.shape[0] != steps.numel():
            raise ValueError(
                "target batch size must equal diffusion timestep batch size: "
                f"{self.target_classes.shape[0]} != {steps.numel()}"
            )
        tokens_per_sample = hidden.shape[0] // steps.numel()
        selected = (steps >= self.t_min) & (steps <= self.t_max)
        if selected.any():
            sample_latents = latents.view(steps.numel(), tokens_per_sample, -1)[
                selected
            ]  # [B_sel, T, D]
            targets = self.target_classes.to(device=sample_latents.device)[selected]
            ratio_table = torch.ones(
                sample_latents.shape[0],
                self.sae.latent_dim,
                device=sample_latents.device,
            )
            applied = torch.zeros(
                sample_latents.shape[0],
                dtype=torch.bool,
                device=sample_latents.device,
            )
            for (stage, shape), ratio in self.group_ratios.items():
                hit = targets[:, stage] == shape
                if hit.any():
                    multi = hit & applied
                    self.multi_hit_samples += int(multi.sum())
                    # Per-dimension multiplicative merge: opposing roles
                    # cancel (2.0 * 0.5 == 1.0), same-direction roles stack.
                    ratio_table[hit] = ratio_table[hit] * ratio.unsqueeze(0)
                    applied |= hit
            ratio_table.clamp_(min=self.min_ratio, max=self.max_ratio)
            steered = sample_latents.detach() * ratio_table.unsqueeze(1)
            if applied.any():
                self.steered_samples += int(applied.sum())
                self.steered_rows += int(applied.sum()) * tokens_per_sample
                self.delta_sum += float(
                    (steered - sample_latents.detach()).abs().sum().cpu()
                )
            steered_rows = steered.reshape(-1, steered.shape[-1])
            row_selected = selected.repeat_interleave(tokens_per_sample)
            transformed[row_selected] = self.sae.decode(steered_rows)
            latents[row_selected] = steered_rows
        return transformed, latents


__all__ = ["MultiplicativeSteeringWrapper", "parse_group_key"]
