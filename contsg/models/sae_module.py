"""Pure Top-K sparse autoencoder used by the retained SAE experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Top-K SAE with standardized inputs and dead-feature auxiliary loss.

    This is the complete architecture used by the retained Top-K=48 checkpoints.
    It intentionally contains only the reconstruction and dead-feature objectives.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        topk_k: int,
        aux_lambda: float = 1e-4,
        normalize_decoder: bool = True,
        dead_tolerance: int = 1000,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if not 1 <= topk_k <= latent_dim:
            raise ValueError("topk_k must be in [1, latent_dim]")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.topk_k = topk_k
        self.aux_lambda = aux_lambda
        self.normalize_decoder = normalize_decoder
        self.dead_tolerance = dead_tolerance

        self.W_enc = nn.Parameter(torch.empty(input_dim, latent_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))
        self.W_dec = nn.Parameter(torch.empty(latent_dim, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("mu_h", torch.zeros(input_dim))
        self.register_buffer("sigma_h", torch.ones(input_dim))
        self.register_buffer("inactive_steps", torch.zeros(latent_dim, dtype=torch.long))
        self.epsilon = 1e-6
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        if self.normalize_decoder:
            self.normalize_decoder_weights()

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_hat = (hidden - self.mu_h) / (self.sigma_h + self.epsilon)
        pre_acts = F.relu((hidden_hat - self.b_dec) @ self.W_enc + self.b_enc)
        values, _ = torch.topk(pre_acts, self.topk_k, dim=-1)
        threshold = values[..., -1:]
        return pre_acts * (pre_acts >= threshold)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_hat = latents @ self.W_dec + self.b_dec
        return hidden_hat * (self.sigma_h + self.epsilon) + self.mu_h

    def forward(
        self, hidden: torch.Tensor, return_dead_info: bool = False
    ) -> tuple[torch.Tensor, ...]:
        hidden_hat = (hidden - self.mu_h) / (self.sigma_h + self.epsilon)
        latents = self.encode(hidden)
        hidden_hat_recon = latents @ self.W_dec + self.b_dec
        hidden_recon = hidden_hat_recon * (self.sigma_h + self.epsilon) + self.mu_h
        if return_dead_info:
            residual = hidden_hat - hidden_hat_recon
            active_mask = latents > 0
            return hidden_recon, latents, residual, active_mask, hidden_hat, hidden_hat_recon
        return hidden_recon, latents

    def compute_loss(
        self, hidden: torch.Tensor, update_dead_stats: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, residual, active_mask, hidden_hat, hidden_hat_recon = self.forward(
            hidden, return_dead_info=True
        )
        reconstruction_loss = F.mse_loss(hidden_hat_recon, hidden_hat)

        if update_dead_stats:
            with torch.no_grad():
                self.inactive_steps.add_(1)
                self.inactive_steps[active_mask.any(dim=0)] = 0

        dead_indices = torch.where(self.inactive_steps >= self.dead_tolerance)[0]
        auxiliary_loss = hidden.new_zeros(())
        if dead_indices.numel() > 0:
            residual_target = residual.detach()
            dead_pre_acts = F.relu(
                residual_target @ self.W_enc[:, dead_indices] + self.b_enc[dead_indices]
            )
            k_aux = min(self.topk_k, dead_pre_acts.shape[-1])
            if k_aux > 0:
                values, indices = torch.topk(dead_pre_acts, k_aux, dim=-1)
                dead_latents = torch.zeros_like(dead_pre_acts)
                dead_latents.scatter_(-1, indices, values)
                dead_recon = dead_latents @ self.W_dec[dead_indices]
                auxiliary_loss = F.mse_loss(dead_recon, residual_target)

        total_loss = reconstruction_loss + self.aux_lambda * auxiliary_loss
        return total_loss, reconstruction_loss, auxiliary_loss

    @torch.no_grad()
    def normalize_decoder_weights(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def update_statistics(self, activations: torch.Tensor) -> None:
        if activations.ndim != 2 or activations.shape[1] != self.input_dim:
            raise ValueError(
                f"expected activations shaped (N, {self.input_dim}), got {tuple(activations.shape)}"
            )
        self.mu_h.copy_(activations.mean(dim=0))
        self.sigma_h.copy_(activations.std(dim=0, unbiased=False).clamp_min(1e-6))

    @torch.no_grad()
    def resample_dead_neurons(self, activations: torch.Tensor) -> int:
        dead_indices = torch.where(self.inactive_steps >= self.dead_tolerance)[0]
        if dead_indices.numel() == 0:
            return 0

        hidden_hat = (activations - self.mu_h) / (self.sigma_h + self.epsilon)
        recon, _ = self(activations)
        recon_hat = (recon - self.mu_h) / (self.sigma_h + self.epsilon)
        probabilities = (hidden_hat - recon_hat).pow(2).sum(dim=-1)
        if probabilities.sum() <= 0:
            probabilities = torch.ones_like(probabilities)
        sample_ids = torch.multinomial(
            probabilities, dead_indices.numel(), replacement=dead_indices.numel() > len(activations)
        )
        directions = F.normalize(hidden_hat[sample_ids], dim=-1)
        encoder_norm = self.W_enc.norm(dim=0).mean().clamp_min(1e-8)
        self.W_enc[:, dead_indices] = directions.t() * encoder_norm
        self.W_dec[dead_indices] = directions
        self.b_enc[dead_indices] = 0
        self.inactive_steps[dead_indices] = 0
        if self.normalize_decoder:
            self.normalize_decoder_weights()
        return int(dead_indices.numel())

    @torch.no_grad()
    def compute_eval_metrics(
        self, data_loader: Iterable, device: torch.device, max_batches: int = 100
    ) -> Dict[str, float | int]:
        was_training = self.training
        self.eval()
        active_counts = torch.zeros(self.latent_dim, device=device)
        total_samples = 0
        total_squared_error = 0.0
        total_raw_squared_error = 0.0
        total_raw_squared_deviation = 0.0
        total_l0 = 0.0

        for batch_index, batch in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            hidden = batch[0] if isinstance(batch, (tuple, list)) else batch
            hidden = hidden.to(device)
            _, latents, _, _, hidden_hat, hidden_hat_recon = self.forward(
                hidden, return_dead_info=True
            )
            active_counts += (latents > 0).sum(dim=0)
            total_samples += hidden.shape[0]
            total_squared_error += F.mse_loss(
                hidden_hat_recon, hidden_hat, reduction="sum"
            ).item()
            total_raw_squared_error += F.mse_loss(
                self.decode(latents), hidden, reduction="sum"
            ).item()
            total_raw_squared_deviation += (hidden - self.mu_h).pow(2).sum().item()
            total_l0 += (latents > 0).sum().item()

        active_features = int((active_counts > 0).sum().item())
        frequencies = active_counts / max(total_samples, 1)
        distribution = frequencies / frequencies.sum().clamp_min(1e-8)
        distribution = distribution[distribution > 0]
        entropy = float(-(distribution * distribution.log()).sum().item())
        max_entropy = math.log(self.latent_dim) if self.latent_dim > 1 else 1.0
        top_n = min(10, self.latent_dim)
        top10_coverage = float(
            frequencies.topk(top_n).values.sum().item() / frequencies.sum().clamp_min(1e-8).item()
        )
        if was_training:
            self.train()
        return {
            "recon_mse": round(
                total_squared_error / max(total_samples * self.input_dim, 1), 8
            ),
            "raw_recon_mse": round(
                total_raw_squared_error / max(total_samples * self.input_dim, 1), 8
            ),
            "explained_variance": round(
                1.0 - total_raw_squared_error / max(total_raw_squared_deviation, 1e-12), 8
            ),
            "l0": round(total_l0 / max(total_samples, 1), 6),
            "dead_ratio": round(1.0 - active_features / self.latent_dim, 6),
            "active_neuron_count": active_features,
            "activation_entropy": round(entropy, 6),
            "normalized_entropy": round(entropy / max_entropy, 6),
            "max_neuron_freq": round(float(frequencies.max().item()), 6),
            "top10_coverage": round(top10_coverage, 6),
            "total_samples_evaluated": total_samples,
        }
__all__ = ["TopKSparseAutoencoder"]