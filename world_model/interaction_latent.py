from __future__ import annotations

from flax import linen as nn
import jax.numpy as jp


class InteractionLatentEncoder(nn.Module):
    """ψ encoder: predicts low-dim interaction latent z from short history."""

    latent_dim: int = 6
    hidden_size: int = 64

    @nn.compact
    def __call__(self, obs_hist: jp.ndarray, act_hist: jp.ndarray) -> jp.ndarray:
        x = jp.concatenate([obs_hist.reshape(-1), act_hist.reshape(-1)], axis=-1)
        x = nn.swish(nn.Dense(self.hidden_size)(x))
        x = nn.swish(nn.Dense(self.hidden_size)(x))
        z = nn.Dense(self.latent_dim)(x)
        return z


def heuristic_interaction_latent(
    obs_hist: jp.ndarray,
    act_hist: jp.ndarray,
    latent_dim: int,
) -> jp.ndarray:
    """Fallback non-parametric ψ for runtime when encoder params are unavailable."""

    obs_energy = jp.mean(jp.square(obs_hist))
    act_energy = jp.mean(jp.square(act_hist))
    delta = obs_hist[-1] - obs_hist[0] if obs_hist.shape[0] > 1 else obs_hist[-1]
    delta_energy = jp.mean(jp.square(delta))
    base = jp.array([obs_energy, act_energy, delta_energy], dtype=obs_hist.dtype)
    if latent_dim <= 3:
        return base[:latent_dim]
    return jp.pad(base, (0, latent_dim - 3))
