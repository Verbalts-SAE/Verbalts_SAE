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
        active_only: bool = False,
        allowed_dims: Sequence[int] | None = None,
        batch_invariant: bool = False,
        objective_head_indices: Sequence[int] | None = None,
        objective_head_weights: Sequence[float] | None = None,
        objective_class_weights: Sequence[float] | None = None,
        guidance_t_range: Tuple[int, int] | None = None,
        allowed_tokens: Sequence[int] | None = None,
        diagnostic_trace: bool = False,
        selection_score: str = "gradient",
        preserve_residual: bool = False,
    ) -> None:
        super().__init__(sae, t_range)
        if selection_score not in ("gradient", "applied"):
            raise ValueError("selection_score must be gradient or applied")
        self.selection_score = selection_score
        self.preserve_residual = preserve_residual
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
        if guidance_t_range is None:
            guidance_t_range = t_range
        guidance_t_range = tuple(int(value) for value in guidance_t_range)
        # An inverted range (min > max) disables the guidance step: the SAE
        # reconstruction stays active over t_min..t_max but the selected
        # guidance mask is empty.
        if len(guidance_t_range) != 2:
            raise ValueError("guidance_t_range must be a (minimum, maximum) pair")
        if guidance_t_range[0] < self.t_min or guidance_t_range[1] > self.t_max:
            raise ValueError("guidance_t_range must be contained in the SAE reconstruction range")
        self.guidance_t_min, self.guidance_t_max = guidance_t_range
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
        if allowed_tokens is not None:
            tokens_per_sample = getattr(classifier, "tokens_per_sample", None)
            if tokens_per_sample is None:
                raise TypeError("allowed_tokens requires classifier.tokens_per_sample")
            tokens = tuple(dict.fromkeys(int(token) for token in allowed_tokens))
            if not tokens or min(tokens) < 0 or max(tokens) >= int(tokens_per_sample):
                raise ValueError(
                    f"allowed_tokens out of range for {tokens_per_sample} tokens per sample"
                )
            token_mask = torch.zeros(int(tokens_per_sample), dtype=torch.float32)
            token_mask[list(tokens)] = 1.0
            self.register_buffer("allowed_token_mask", token_mask)
        else:
            self.register_buffer(
                "allowed_token_mask", torch.empty((0,), dtype=torch.float32)
            )
        self.classifier = classifier
        self.diagnostic_trace = diagnostic_trace
        self.trace_records = []
        self.eta = float(eta)
        self.rel_cap = float(rel_cap)
        self.max_step = float(max_step)
        self.iters = int(iters)
        self.topk = int(topk)
        self.topk_mode = topk_mode
        # Restrict the guidance step to latents that are ACTIVE in the current
        # token (z > 1e-6).  The classifier standardizes by input_std, so
        # dead latents (z=0, tiny sigma) produce exploding gradients that
        # dominate the per-token top-k ranking while their sigma cap keeps
        # their actual move negligible; selecting them wastes the top-k budget
        # and perturbs decoder directions that carry no information.
        self.active_only = bool(active_only)
        # Historical guidance averages over the entire batch, so its effective
        # per-sample dose scales as 1 / batch_size.  Keep that behavior by
        # default for reproducibility; new evaluations can sum per-sample
        # objectives to make eta independent of generation batch size.
        self.batch_invariant = bool(batch_invariant)
        # Keep the historical all-head reduction as the default execution
        # path.  A non-empty subset zeros inactive head losses while retaining
        # the original head-count denominator, so masking a head removes its
        # contribution without renormalizing (and increasing) the remaining
        # heads' dose.
        if objective_head_indices is None:
            self.register_buffer(
                "objective_head_indices", torch.empty((0,), dtype=torch.long)
            )
        else:
            head_indices = tuple(dict.fromkeys(int(index) for index in objective_head_indices))
            if not head_indices or min(head_indices) < 0:
                raise ValueError("objective_head_indices must be non-empty and non-negative")
            self.register_buffer(
                "objective_head_indices", torch.tensor(head_indices, dtype=torch.long)
            )
        # Optional per-head multiplicative weights applied AFTER the 0/1
        # objective-head mask so per-stage dose can be graded (e.g. a weak
        # beginning/end dose plus a full middle dose) instead of binary.
        if objective_head_weights is None:
            self.register_buffer(
                "objective_head_weights", torch.empty((0,), dtype=torch.float32)
            )
        else:
            weights = tuple(float(value) for value in objective_head_weights)
            if not weights or min(weights) < 0.0:
                raise ValueError("objective_head_weights must be non-empty and non-negative")
            self.register_buffer(
                "objective_head_weights", torch.tensor(weights, dtype=torch.float32)
            )
        # Optional per-class multiplicative weights applied to the objective
        # loss of each segment head based on the TARGET class of that segment
        # (e.g. weighting PVC targets x2 to compensate a hard class).  The
        # weight tensor is indexed by the gathered target class per head, so
        # its length must match the classifier's class count.
        if objective_class_weights is None:
            self.register_buffer(
                "objective_class_weights", torch.empty((0,), dtype=torch.float32)
            )
        else:
            class_weights = tuple(float(value) for value in objective_class_weights)
            if not class_weights or min(class_weights) < 0.0:
                raise ValueError("objective_class_weights must be non-empty and non-negative")
            self.register_buffer(
                "objective_class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
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
        # "dynamic": per-token top-k of the current absolute gradient (the
        # state's own selection set).  topk=0 keeps the dense gradient.
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
        self._reset_guidance_audit()

    def _reset_guidance_audit(self) -> None:
        """Reset guidance-only dose diagnostics without changing the intervention."""

        self.trace_records = []
        self.guidance_rows = 0
        self.guidance_gradient_abs_sum = 0.0
        self.guidance_step_sum = 0.0
        self.guidance_pre_cap_step_sum = 0.0
        self.guidance_cap_hits = 0
        self.guidance_selected_cap_rows = 0
        self.guidance_selected_cap_hits = 0
        self.guidance_applied_rows = 0
        self.guidance_applied_nonzero = 0
        self.guidance_applied_max_nonzero_per_token = 0
        self.guidance_applied_step_sum = 0.0
        self.guidance_applied_standardized_step_sum = 0.0

    def reset_audit(self) -> None:
        """Reset both SAE-hook and guidance-dose audit counters."""

        super().reset_audit()
        self._reset_guidance_audit()

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

    def audit(self) -> dict[str, float | int | bool]:
        report = super().audit()
        report["guidance_rows"] = self.guidance_rows
        report["guidance_raw_gradient_mean_abs"] = (
            self.guidance_gradient_abs_sum / max(self.guidance_rows, 1)
        )
        report["guidance_mean_abs_step"] = self.guidance_step_sum / max(
            self.guidance_rows, 1
        )
        report["guidance_pre_cap_mean_abs_step"] = (
            self.guidance_pre_cap_step_sum / max(self.guidance_rows, 1)
        )
        report["guidance_cap_fraction"] = self.guidance_cap_hits / max(
            self.guidance_rows, 1
        )
        report["guidance_selected_cap_fraction"] = (
            self.guidance_selected_cap_hits
            / max(self.guidance_selected_cap_rows, 1)
        )
        report["guidance_applied_mean_abs_step"] = (
            self.guidance_applied_step_sum / max(self.guidance_applied_rows, 1)
        )
        report["guidance_applied_nonzero_fraction"] = (
            self.guidance_applied_nonzero / max(self.guidance_applied_rows, 1)
        )
        report["guidance_topk"] = self.topk
        report["guidance_active_only"] = self.active_only
        report["guidance_applied_max_nonzero_per_token"] = (
            self.guidance_applied_max_nonzero_per_token
        )
        report["guidance_applied_nonzero_mean_abs_step"] = (
            self.guidance_applied_step_sum / max(self.guidance_applied_nonzero, 1)
        )
        report["guidance_applied_mean_abs_step_over_sigma"] = (
            self.guidance_applied_standardized_step_sum
            / max(self.guidance_applied_rows, 1)
        )
        report["batch_invariant"] = self.batch_invariant
        report["guidance_t_min"] = self.guidance_t_min
        report["guidance_t_max"] = self.guidance_t_max
        report["guidance_token_count"] = (
            int(self.allowed_token_mask.sum())
            if self.allowed_token_mask.numel() > 0
            else int(getattr(self.classifier, "tokens_per_sample", 0))
        )
        return report

    def forward(
        self, hidden: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed, latents = super().forward(hidden, diffusion_step)
        if self.preserve_residual:
            transformed = hidden.clone()
        steps = torch.as_tensor(diffusion_step, device=hidden.device).long().flatten()
        if self.target_classes.shape[0] != steps.numel():
            raise ValueError(
                "target batch size must equal diffusion timestep batch size: "
                f"{self.target_classes.shape[0]} != {steps.numel()}"
            )
        tokens_per_sample = hidden.shape[0] // steps.numel()
        # SAE reconstruction remains active over self.t_min..self.t_max; only
        # the causal guidance delta is restricted by this independent window.
        selected = (steps >= self.guidance_t_min) & (steps <= self.guidance_t_max)
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
                        losses = -(target_log_probs * weights)
                    else:
                        losses = -target_log_probs
                    if self.objective_head_indices.numel() > 0:
                        if losses.ndim != 2 or int(self.objective_head_indices.max()) >= losses.shape[1]:
                            raise ValueError(
                                "objective head mask/classifier output mismatch: "
                                f"losses={tuple(losses.shape)}, "
                                f"indices={self.objective_head_indices.tolist()}"
                            )
                        head_mask = losses.new_zeros(losses.shape[1])
                        head_mask[self.objective_head_indices] = 1.0
                        losses = losses * head_mask
                    if self.objective_head_weights.numel() > 0:
                        if losses.ndim != 2 or self.objective_head_weights.numel() != losses.shape[1]:
                            raise ValueError(
                                "objective head weights/classifier output mismatch: "
                                f"losses={tuple(losses.shape)}, "
                                f"weights={self.objective_head_weights.tolist()}"
                            )
                        losses = losses * self.objective_head_weights.to(losses.device)
                    if self.objective_class_weights.numel() > 0:
                        if int(targets.max()) >= self.objective_class_weights.numel():
                            raise ValueError(
                                "objective class weights/classifier output mismatch: "
                                f"targets up to {int(targets.max())}, "
                                f"weights={self.objective_class_weights.tolist()}"
                            )
                        class_w = self.objective_class_weights.to(losses.device)[targets]
                        losses = losses * class_w
                    if self.batch_invariant:
                        loss = losses.reshape(losses.shape[0], -1).mean(dim=1).sum()
                    else:
                        loss = losses.mean()
                    (grad,) = torch.autograd.grad(loss, leaf)
                self.guidance_gradient_abs_sum += float(
                    grad.detach().abs().sum().cpu()
                )
                scaled_grad = torch.nan_to_num(self.eta * grad)
                cap_view = caps.view(1, 1, -1)
                delta = scaled_grad.clamp(
                    -caps.view(1, 1, -1), caps.view(1, 1, -1)
                )
                self.guidance_pre_cap_step_sum += float(
                    scaled_grad.detach().abs().sum().cpu()
                )
                self.guidance_cap_hits += int(
                    (scaled_grad.detach().abs() > cap_view).sum().cpu()
                )
                if self.allowed_mask.numel() > 0:
                    delta = delta * self.allowed_mask.to(delta.device)
                if self.allowed_token_mask.numel() > 0:
                    delta = delta * self.allowed_token_mask.to(delta.device).view(1, -1, 1)
                if self.active_only:
                    # Dead latents never move under active-only steering.
                    active_mask = sample_latents.detach() > 1e-6
                    delta = delta * active_mask.to(delta.dtype)
                current = torch.clamp_min(leaf.detach() - delta, 0.0)
                self.guidance_rows += int(delta.numel())
                self.guidance_step_sum += float(delta.detach().abs().sum().cpu())
            stepped = current
            selected_mask = torch.ones_like(grad, dtype=torch.bool)
            active_mask = (
                (sample_latents.detach() > 1e-6)
                if self.active_only else None
            )
            if self.topk > 0:
                if self.topk_mode == "global":
                    target_for_mask = targets if targets.ndim == 1 else targets[:, 1]
                    mask = self.global_topk_mask[
                        target_for_mask
                    ].unsqueeze(1)  # [B_sel, 1, D]
                    if self.active_only:
                        mask = mask & active_mask
                    selected_mask = mask.expand_as(stepped)
                    stepped = torch.where(
                        selected_mask, stepped, sample_latents.detach()
                    )
                else:  # dynamic per-token top-k of the current |gradient|
                    gradient_magnitude = grad.detach().abs()
                    if self.selection_score == "applied":
                        gradient_magnitude = (stepped - sample_latents.detach()).abs()
                    if self.active_only:
                        # Exclude dead latents from the ranking so the top-k
                        # budget is spent on latents that can actually move.
                        gradient_magnitude = gradient_magnitude.masked_fill(
                            ~active_mask, float("-inf")
                        )
                        min_active = int(active_mask.sum(-1).min().item())
                        k_eff = min(self.topk, min_active)
                    else:
                        k_eff = self.topk
                    topk_idx = gradient_magnitude.topk(
                        k_eff, dim=-1
                    ).indices  # [B_sel, T, k_eff]
                    mask = torch.zeros_like(gradient_magnitude, dtype=torch.bool)
                    mask.scatter_(-1, topk_idx, True)
                    selected_mask = mask
                    stepped = torch.where(
                        mask, stepped, sample_latents.detach()
                    )
            selected_cap_hits = scaled_grad.detach().abs() > cap_view
            self.guidance_selected_cap_rows += int(selected_mask.sum().cpu())
            self.guidance_selected_cap_hits += int(
                (selected_cap_hits & selected_mask).sum().cpu()
            )
            applied_step = stepped - sample_latents.detach()
            if self.diagnostic_trace:
                with torch.no_grad():
                    before = self.classifier(sample_latents.float()).softmax(-1)
                    after = self.classifier(stepped.float()).softmax(-1)
                    pb = before.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    pa = after.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    # Compare both round trips: encode(decode(z)) is not z,
                    # even without guidance. Keep this observational only.
                    decoded_before = self.sae.decode(sample_latents.detach())
                    decoded_after = self.sae.decode(stepped.detach())
                    roundtrip_before = self.sae.encode(decoded_before)
                    roundtrip_after = self.sae.encode(decoded_after)
                    prb = self.classifier(roundtrip_before.float()).softmax(-1).gather(
                        -1, targets.unsqueeze(-1)).squeeze(-1)
                    pra = self.classifier(roundtrip_after.float()).softmax(-1).gather(
                        -1, targets.unsqueeze(-1)).squeeze(-1)
                    hidden_delta = decoded_after - decoded_before
                    hidden_scale = self.sae.sigma_h + self.sae.epsilon
                    input_hidden = hidden.view(steps.numel(), tokens_per_sample, -1)[selected]
                    batch_rows = selected.nonzero(as_tuple=True)[0].tolist()
                    for j, timestep in enumerate(steps[selected].tolist()):
                        mask_j = selected_mask[j]
                        count = max(int(mask_j.sum()), 1)
                        self.trace_records.append(dict(
                            timestep=timestep, target=targets[j].tolist(),
                            batch_row=batch_rows[j],
                            probability_before=pb[j].tolist(), probability_after=pa[j].tolist(),
                            probability_roundtrip_before=prb[j].tolist(),
                            probability_roundtrip_after=pra[j].tolist(),
                            latent_delta_rmse=float(applied_step[j].square().mean().sqrt()),
                            roundtrip_latent_delta_rmse=float((roundtrip_after[j] - roundtrip_before[j]).square().mean().sqrt()),
                            hidden_delta_rmse=float(hidden_delta[j].square().mean().sqrt()),
                            hidden_delta_standardized_rmse=float((hidden_delta[j] / hidden_scale).square().mean().sqrt()),
                            hidden_delta_relative_l2=float(hidden_delta[j].norm() / decoded_before[j].norm().clamp_min(1e-12)),
                            hidden_reconstruction_rmse=float((decoded_before[j] - input_hidden[j]).square().mean().sqrt()),
                            prediction_before=before[j].argmax(-1).tolist(),
                            selected_cap_fraction=float((selected_cap_hits[j] & mask_j).sum()) / count,
                            selected_dead_fraction=float(((sample_latents[j] <= 1e-6) & mask_j).sum()) / count,
                            selected_relu_clip_fraction=float(((sample_latents[j] - delta[j] < 0) & mask_j).sum()) / count,
                            selected_zero_move_fraction=float(((applied_step[j] == 0) & mask_j).sum()) / count,
                            applied_mean_abs=float(applied_step[j].abs().mean()),
                        ))
            self.guidance_applied_rows += int(applied_step.numel())
            self.guidance_applied_nonzero += int((applied_step != 0).sum().cpu())
            self.guidance_applied_max_nonzero_per_token = max(
                self.guidance_applied_max_nonzero_per_token,
                int((applied_step != 0).sum(dim=-1).max().cpu()),
            )
            self.guidance_applied_step_sum += float(
                applied_step.detach().abs().sum().cpu()
            )
            self.guidance_applied_standardized_step_sum += float(
                (applied_step.detach().abs() / sigma.view(1, 1, -1)).sum().cpu()
            )
            steered_rows = stepped.reshape(-1, stepped.shape[-1])
            row_selected = selected.repeat_interleave(tokens_per_sample)
            transformed[row_selected] = self.sae.decode(steered_rows)
            if self.preserve_residual:
                transformed[row_selected] += hidden[row_selected] - self.sae.decode(
                    sample_latents.reshape(-1, sample_latents.shape[-1]))
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