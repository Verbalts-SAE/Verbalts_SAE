"""Checkpoint loading and timestep-aware latent guidance steering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from contsg.models.sae_module import TopKSparseAutoencoder


def load_sae_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TopKSparseAutoencoder, Tuple[int, int], Mapping[str, Any]]:
    """Load a Top-K SAE and its inclusive diffusion-timestep range."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"state_dict", "t_range", "input_dim", "latent_dim", "topk_k"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"SAE checkpoint {path} is missing keys: {sorted(missing)}")

    t_range = tuple(int(value) for value in checkpoint["t_range"])
    if len(t_range) != 2 or t_range[0] > t_range[1]:
        raise ValueError(f"invalid t_range in {path}: {checkpoint['t_range']}")

    model = TopKSparseAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        topk_k=int(checkpoint["topk_k"]),
        aux_lambda=float(checkpoint.get("aux_lambda", 1e-4)),
        normalize_decoder=bool(checkpoint.get("normalize_decoder", True)),
        dead_tolerance=int(checkpoint.get("dead_tolerance", 1000)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, (t_range[0], t_range[1]), checkpoint


class TimestepSAEWrapper(nn.Module):
    """Apply one SAE only to rows belonging to its inclusive timestep window."""

    def __init__(self, sae: TopKSparseAutoencoder, t_range: Tuple[int, int]) -> None:
        super().__init__()
        self.sae = sae
        self.t_min, self.t_max = t_range
        self.reset_audit()

    def reset_audit(self) -> None:
        """Reset counters used to prove that the configured hook was exercised."""
        self.forward_calls = 0
        self.selected_rows = 0
        self.total_rows = 0
        self.reconstruction_squared_error = 0.0
        self.reconstruction_elements = 0

    def audit(self) -> dict[str, float | int]:
        return {
            "forward_calls": self.forward_calls,
            "selected_rows": self.selected_rows,
            "total_rows": self.total_rows,
            "selected_fraction": self.selected_rows / max(self.total_rows, 1),
            "reconstruction_mse": self.reconstruction_squared_error
            / max(self.reconstruction_elements, 1),
        }

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        if steps.numel() == 0 or hidden.shape[0] % steps.numel() != 0:
            raise ValueError(
                "flattened activation count must be divisible by diffusion timestep count"
            )
        tokens_per_sample = hidden.shape[0] // steps.numel()
        row_steps = steps.repeat_interleave(tokens_per_sample)
        selected = (row_steps >= self.t_min) & (row_steps <= self.t_max)
        self.forward_calls += 1
        self.total_rows += int(hidden.shape[0])
        self.selected_rows += int(selected.sum())

        transformed = hidden.clone()
        latents = hidden.new_zeros((hidden.shape[0], self.sae.latent_dim))
        if selected.any():
            reconstructed, selected_latents = self.sae(hidden[selected])
            self.reconstruction_squared_error += float(
                (reconstructed.detach() - hidden[selected].detach()).pow(2).sum().cpu()
            )
            self.reconstruction_elements += reconstructed.numel()
            transformed[selected] = reconstructed
            latents[selected] = selected_latents
        return transformed, latents


class LatentClassifierGuidanceWrapper(TimestepSAEWrapper):
    """Per-sample classifier-guidance on SAE latents within the t-window.

    At every in-window diffusion step the SAE latents of each sample are
    shifted by one gradient-ascent step on the log-probability of its own
    target class(es) under a trained latent classifier:
    ``z_new = relu(z - eta * grad_z mean_stage(-log p_target(z)))``.

    Unlike fixed-direction steering, the perturbation adapts to the current
    latent state of each sample.
    """

    def __init__(
        self,
        sae: TopKSparseAutoencoder,
        t_range: Tuple[int, int],
        classifier: nn.Module,
        eta: float,
        rel_cap: float = 1.0,
        max_step: float = 0.5,
        iters: int = 1,
        adaptive_gamma: float = 0.0,
        full_strength_shapes: str = "",
        topk: int = 0,
        topk_mode: str = "global",
        allowed_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__(sae, t_range)
        if eta <= 0.0:
            raise ValueError("guidance eta must be positive")
        if rel_cap <= 0.0 or max_step <= 0.0:
            raise ValueError("guidance caps must be positive")
        if iters < 1:
            raise ValueError("guidance iters must be at least 1")
        if topk < 0:
            raise ValueError("topk must be >= 0 (0 = dense)")
        if topk > int(sae.latent_dim):
            raise ValueError("topk must not exceed the SAE latent dimension")
        if topk_mode not in ("global", "dynamic"):
            raise ValueError(f"unknown topk_mode {topk_mode!r}")
        # Optional hard restriction of the guidance step to a fixed subset
        # of latent dimensions (e.g. a frequency band or a hand-picked
        # feature set).  None keeps the full dense gradient.  When set, the
        # gradient is masked AFTER the sigma cap so only these dims move.
        if allowed_dims is not None:
            dims = [int(d) for d in allowed_dims]
            if not dims or min(dims) < 0 or max(dims) >= int(sae.latent_dim):
                raise ValueError(
                    f"allowed_dims out of range for latent_dim {sae.latent_dim}"
                )
            mask = torch.zeros(int(sae.latent_dim), dtype=torch.float32)
            mask[dims] = 1.0
            self.register_buffer("allowed_mask", mask)
        else:
            self.register_buffer(
                "allowed_mask", torch.empty((0,), dtype=torch.float32)
            )
        self.classifier = classifier
        self.eta = float(eta)
        self.rel_cap = float(rel_cap)
        self.max_step = float(max_step)
        self.iters = int(iters)
        self.topk = int(topk)
        self.topk_mode = topk_mode
        # Confidence-weighted guidance: per-segment loss is scaled by
        # (1 - p_target)^adaptive_gamma so segments the classifier already
        # gets right receive little perturbation.  Segments whose target
        # shape is saturated (e.g. "nothing" at ~98%) then stop consuming
        # MSE budget while uncertain segments get the full gradient.
        if adaptive_gamma < 0.0:
            raise ValueError("guidance adaptive_gamma must be >= 0")
        self.adaptive_gamma = float(adaptive_gamma)
        # Shape-aware hybrid: segments whose TARGET shape is listed in
        # full_strength_shapes keep weight 1.0 regardless of confidence
        # (sag / double peaks carry most of the guidance gains), while
        # other shapes use the adaptive weighting.  Empty string keeps the
        # pure adaptive behaviour.
        shape_index = {"nothing": 0, "single peak": 1, "double peaks": 2, "sag": 3}
        parsed = [tok.strip() for tok in full_strength_shapes.split(",") if tok.strip()]
        for name in parsed:
            if name not in shape_index:
                raise ValueError(f"unknown full-strength shape {name!r}")
        self.full_strength_idx = tuple(shape_index[name] for name in parsed)
        # Optional sparse truncation of the guidance step.
        # "global": restrict to the top-k latents of the classifier's
        # per-shape composite weight (a FIXED feature set per shape).
        # "dynamic": per-token top-k of the current gradient (the state's
        # own selection set).  topk=0 keeps the dense gradient.
        if self.topk > 0 and self.topk_mode == "global":
            win = classifier.token_projection.weight.detach()
            wout = classifier.output.weight.detach()
            # The SAEClassClassifier shares ONE 4-way head across the three
            # stages, so the composite per-shape direction is [4, D].
            composite = (wout @ win)  # [4, D]
            topk_idx = composite.abs().topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(composite, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            self.register_buffer("global_topk_mask", mask)  # [4, D]
        else:
            self.register_buffer(
                "global_topk_mask", torch.empty((0, 0), dtype=torch.bool)
            )
        initial = torch.empty((0,), dtype=torch.long)
        self.register_buffer("target_classes", initial)
        self.guidance_rows = 0
        self.guidance_step_sum = 0.0

    def set_targets(self, target_classes: torch.Tensor) -> None:
        """Set targets shaped ``(batch,)`` or legacy ``(batch, heads)``."""

        values = torch.as_tensor(target_classes, dtype=torch.long)
        if values.ndim not in (1, 2):
            raise ValueError(
                "target_classes must have shape (batch,) or (batch, heads), got "
                f"{tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            raise ValueError("target_classes must contain at least one sample")
        output = getattr(self.classifier, "output", None)
        num_classes = getattr(output, "out_features", None)
        if num_classes is None:
            raise TypeError("latent classifier must expose output.out_features")
        if values.min() < 0 or values.max() >= int(num_classes):
            raise ValueError(f"target class indices must be in [0, {num_classes - 1}]")
        self.target_classes = values.detach().clone()

    def audit(self) -> dict[str, float | int]:
        report = super().audit()
        report["guidance_rows"] = self.guidance_rows
        report["guidance_mean_abs_step"] = self.guidance_step_sum / max(
            self.guidance_rows, 1
        )
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
            # Clamp the per-latent step so the standardized-space shift stays
            # within rel_cap * sigma for every latent.  The classifier divides
            # by input_std (floor 1e-6), so raw gradients of near-dead latents
            # explode; capping by their own sigma keeps them ~frozen while
            # max_step bounds the absolute move of high-variance latents.
            sigma = self.classifier.input_std.to(sample_latents.device)  # [D]
            caps = torch.minimum(
                self.rel_cap * sigma, torch.full_like(sigma, self.max_step)
            )
            targets = self.target_classes.to(device=sample_latents.device)[selected]
            current = sample_latents.detach()
            for _ in range(self.iters):
                with torch.enable_grad():
                    leaf = current.requires_grad_(True)
                    logits = self.classifier(leaf.float())
                    log_probs = logits.log_softmax(dim=-1)
                    if logits.ndim == 2 and targets.ndim == 1:
                        target_log_probs = log_probs.gather(
                            -1, targets.unsqueeze(-1)
                        ).squeeze(-1)
                    elif logits.ndim == 3 and targets.ndim == 2:
                        if logits.shape[1] != targets.shape[1]:
                            raise ValueError(
                                "classifier head count does not match target head count"
                            )
                        target_log_probs = log_probs.gather(
                            -1, targets.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        raise ValueError(
                            "classifier output/target mismatch: "
                            f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
                        )
                    if self.adaptive_gamma > 0.0:
                        probs = target_log_probs.detach().exp()
                        weights = (1.0 - probs).clamp_min(0.0) ** self.adaptive_gamma
                        if self.full_strength_idx:
                            full_tensor = torch.tensor(
                                self.full_strength_idx, device=targets.device
                            )
                            full_mask = torch.isin(targets, full_tensor)
                            weights = torch.where(
                                full_mask, torch.ones_like(weights), weights
                            )
                        loss = -(target_log_probs * weights).mean()
                    else:
                        loss = -target_log_probs.mean()
                    (grad,) = torch.autograd.grad(loss, leaf)
                delta = torch.nan_to_num(self.eta * grad).clamp(
                    -caps.view(1, 1, -1), caps.view(1, 1, -1)
                )
                if self.allowed_mask.numel() > 0:
                    delta = delta * self.allowed_mask.to(delta.device)
                current = torch.clamp_min(leaf.detach() - delta, 0.0)
                self.guidance_rows += int(delta.numel())
                self.guidance_step_sum += float(delta.detach().abs().sum().cpu())
            stepped = current
            if self.topk > 0:
                if self.topk_mode == "global":
                    target_for_mask = targets if targets.ndim == 1 else targets[:, 1]
                    mask = self.global_topk_mask[
                        target_for_mask
                    ].unsqueeze(1)  # [B_sel, 1, D]
                    stepped = torch.where(
                        mask.expand_as(stepped), stepped, sample_latents.detach()
                    )
                else:  # dynamic per-token top-k of |stepped - original|
                    step_mag = (stepped - sample_latents.detach()).abs()
                    topk_idx = step_mag.topk(
                        self.topk, dim=-1
                    ).indices  # [B_sel, T, k]
                    mask = torch.zeros_like(step_mag, dtype=torch.bool)
                    mask.scatter_(-1, topk_idx, True)
                    stepped = torch.where(
                        mask, stepped, sample_latents.detach()
                    )
            steered_rows = stepped.reshape(-1, stepped.shape[-1])
            row_selected = selected.repeat_interleave(tokens_per_sample)
            transformed[row_selected] = self.sae.decode(steered_rows)
            latents[row_selected] = steered_rows
        return transformed, latents


class FixedDirectionAdditiveWrapper(TimestepSAEWrapper):
    """Additive steering along a FIXED per-shape direction.

    Control operator for the IG-vs-guidance comparison.  Unlike
    :class:`LatentClassifierGuidanceWrapper` the direction is static (computed
    once offline, e.g. an IG direction on a linear classifier); only the
    middle-stage target of each sample selects which of the 4 shape directions
    applies.  The perturbation is identical at every in-window diffusion step
    and for every token.

    ``mode`` selects the coordinate treatment of the fixed direction:

    * ``"raw"``    — raw z-unit direction (dead latents dominate);
    * ``"rms"``    — sign(standardized dir) * active_rms (the historical
      additive recipe);
    * ``"sigcap"`` — standardized unit direction with the same per-latent
      sigma cap as the guidance wrapper (its calibration, but static).
    """

    def __init__(
        self,
        sae: TopKSparseAutoencoder,
        t_range: Tuple[int, int],
        directions_std: torch.Tensor,
        directions_raw: torch.Tensor,
        active_rms: torch.Tensor,
        sigma: torch.Tensor,
        strength: float,
        mode: str = "rms",
        rel_cap: float = 1.0,
        max_step: float = 0.5,
    ) -> None:
        super().__init__(sae, t_range)
        if mode not in ("raw", "rms", "sigcap"):
            raise ValueError(f"unknown fixed-direction mode {mode!r}")
        if strength <= 0.0:
            raise ValueError("strength must be positive")
        self.register_buffer("directions_std", torch.as_tensor(
            directions_std, dtype=torch.float32).clone())
        self.register_buffer("directions_raw", torch.as_tensor(
            directions_raw, dtype=torch.float32).clone())
        self.register_buffer("active_rms", torch.as_tensor(
            active_rms, dtype=torch.float32).clone())
        self.register_buffer("sigma", torch.as_tensor(
            sigma, dtype=torch.float32).clone())
        self.mode = mode
        self.strength = float(strength)
        self.rel_cap = float(rel_cap)
        self.max_step = float(max_step)
        initial = torch.empty((0, 3), dtype=torch.long)
        self.register_buffer("target_classes", initial)
        self.steering_rows = 0
        self.step_sum = 0.0

    def set_targets(self, target_classes: torch.Tensor) -> None:
        values = torch.as_tensor(target_classes, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(
                "target_classes must have shape (batch, 3), got "
                f"{tuple(values.shape)}"
            )
        if values.min() < 0 or values.max() >= self.directions_std.shape[0]:
            raise ValueError("target class indices out of range")
        self.target_classes = values.detach().clone()

    def audit(self) -> dict[str, float | int]:
        report = super().audit()
        report["steering_rows"] = self.steering_rows
        report["mean_abs_step"] = self.step_sum / max(self.steering_rows, 1)
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
            middle_targets = targets[:, 1]  # [B_sel] middle-stage shape index
            d_std = self.directions_std[middle_targets]  # [B_sel, D]
            if self.mode == "raw":
                d_raw = self.directions_raw[middle_targets]
                step = self.strength * d_raw
            elif self.mode == "rms":
                step = self.strength * torch.sign(d_std) * self.active_rms
            else:  # sigcap
                caps = torch.minimum(
                    self.rel_cap * self.sigma.to(sample_latents.device),
                    torch.full_like(self.sigma.to(sample_latents.device), self.max_step),
                )
                step = (self.strength * d_std).clamp(
                    -caps.view(1, -1), caps.view(1, -1)
                )
            stepped = torch.clamp_min(
                sample_latents.detach() + step.unsqueeze(1), 0.0
            )
            self.steering_rows += int(step.numel())
            self.step_sum += float(step.abs().sum().cpu())
            steered_rows = stepped.reshape(-1, stepped.shape[-1])
            row_selected = selected.repeat_interleave(tokens_per_sample)
            transformed[row_selected] = self.sae.decode(steered_rows)
            latents[row_selected] = steered_rows
        return transformed, latents


class HGuidanceWrapper(nn.Module):
    """Classifier guidance applied DIRECTLY on the raw h residual (no SAE).

    Control operator answering "do we need the SAE at all?".  The same
    per-sample gradient-ascent protocol runs in the 64-dim residual space:
    ``h_new = h - eta * grad_h mean_stage(-log p_target(h))`` with the same
    per-dim sigma cap.  The classifier standardizes its input internally;
    the chain rule converts gradients back to raw h units.
    """

    def __init__(
        self,
        classifier: nn.Module,
        t_range: Tuple[int, int],
        eta: float,
        rel_cap: float = 1.0,
        max_step: float = 0.5,
    ) -> None:
        super().__init__()
        if eta <= 0.0:
            raise ValueError("guidance eta must be positive")
        self.classifier = classifier
        self.eta = float(eta)
        self.rel_cap = float(rel_cap)
        self.max_step = float(max_step)
        self.t_min, self.t_max = t_range
        initial = torch.empty((0, 3), dtype=torch.long)
        self.register_buffer("target_classes", initial)
        self.forward_calls = 0
        self.guidance_rows = 0
        self.step_sum = 0.0

    def set_targets(self, target_classes: torch.Tensor) -> None:
        values = torch.as_tensor(target_classes, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(
                "target_classes must have shape (batch, 3), got "
                f"{tuple(values.shape)}"
            )
        self.target_classes = values.detach().clone()

    def audit(self) -> dict[str, float | int]:
        return {
            "forward_calls": self.forward_calls,
            "guidance_rows": self.guidance_rows,
            "mean_abs_step": self.step_sum / max(self.guidance_rows, 1),
        }

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        if steps.numel() == 0 or hidden.shape[0] % steps.numel() != 0:
            raise ValueError(
                "flattened activation count must be divisible by diffusion timestep count"
            )
        tokens_per_sample = hidden.shape[0] // steps.numel()
        self.forward_calls += 1
        selected = (steps >= self.t_min) & (steps <= self.t_max)
        transformed = hidden.clone()
        if selected.any():
            rows = hidden.view(steps.numel(), tokens_per_sample, -1)[selected]
            targets = self.target_classes.to(device=rows.device)[selected]
            with torch.enable_grad():
                leaf = rows.detach().requires_grad_(True)
                logits = self.classifier(leaf)  # [B_sel, 3, 4]
                log_probs = logits.log_softmax(dim=-1)
                target_log_probs = log_probs.gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                (grad,) = torch.autograd.grad(
                    -target_log_probs.mean(), leaf
                )  # standardized space
            sigma = self.classifier.input_std.to(rows.device)
            grad_h = grad / sigma  # chain rule back to raw h units
            caps = torch.minimum(
                self.rel_cap * sigma, torch.full_like(sigma, self.max_step)
            )
            delta = torch.nan_to_num(self.eta * grad_h).clamp(
                -caps.view(1, 1, -1), caps.view(1, 1, -1)
            )
            steered = leaf.detach() - delta
            self.guidance_rows += int(delta.numel())
            self.step_sum += float(delta.detach().abs().sum().cpu())
            row_selected = selected.repeat_interleave(tokens_per_sample)
            transformed[row_selected] = steered.reshape(-1, steered.shape[-1])
        return transformed, None


class CompositeGuidanceWrapper(nn.Module):
    """Sequentially apply several guidance components on their own t-windows.

    Each component encodes the current residual with its own SAE, applies its
    per-sample capped gradient-ascent step inside its window, and decodes back.
    The residual stream of one component feeds the next, so overlapping
    windows refine the same layer-1 representation in order.
    """

    def __init__(self, components: Sequence[LatentClassifierGuidanceWrapper]) -> None:
        super().__init__()
        if not components:
            raise ValueError("composite guidance needs at least one component")
        self.components = nn.ModuleList(components)

    def set_targets(self, target_classes: torch.Tensor) -> None:
        for component in self.components:
            component.set_targets(target_classes)

    def reset_audit(self) -> None:
        for component in self.components:
            component.reset_audit()

    def audit(self) -> dict[str, float | int]:
        report: dict[str, float | int] = {"components": len(self.components)}
        for index, component in enumerate(self.components):
            for key, value in component.audit().items():
                report[f"c{index}_{key}"] = value
        return report

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        current = hidden
        for component in self.components:
            current, _ = component(current, diffusion_step)
        return current, None


def attach_sae(
    verbalts_module: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> TimestepSAEWrapper:
    """Attach a retained SAE to residual layer 1 of a VerbalTS module."""
    core = getattr(verbalts_module, "verbalts", verbalts_module)
    if not hasattr(core, "activation_transform"):
        raise TypeError("target is not a compatible VerbalTS module/core")
    if device is None:
        device = next(core.parameters()).device
    sae, t_range, _ = load_sae_checkpoint(checkpoint_path, device=device)
    if sae.input_dim != core.channels:
        raise ValueError(
            f"SAE input_dim={sae.input_dim} does not match VerbalTS channels={core.channels}"
        )
    wrapper = TimestepSAEWrapper(sae, t_range).to(device).eval()
    core.activation_transform = wrapper
    return wrapper


def detach_sae(verbalts_module: nn.Module) -> None:
    """Restore Pure VerbalTS behavior."""
    core = getattr(verbalts_module, "verbalts", verbalts_module)
    if not hasattr(core, "activation_transform"):
        raise TypeError("target is not a compatible VerbalTS module/core")
    core.activation_transform = None


__all__ = [
    "CompositeGuidanceWrapper",
    "LatentClassifierGuidanceWrapper",
    "TimestepSAEWrapper",
    "attach_sae",
    "detach_sae",
    "load_sae_checkpoint",
]