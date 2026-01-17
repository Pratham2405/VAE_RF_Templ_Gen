# losses/vae_loss.py

import torch
import torch.nn.functional as F

def vae_loss(recon, x, mu, logvar, rec_w, kl_w, mask=None, channel_weights=None):
    """
    VAE loss with optional per-channel reconstruction weighting.

    Parameters
    ----------
    recon, x : torch.Tensor of shape [B, C, H, W]
        Reconstructed and target tensors
    mu, logvar : torch.Tensor
        Latent distribution parameters
    rec_w, kl_w : float
        Overall reconstruction and KL weights
    mask : torch.Tensor of shape [B, H, W] or None
        Optional mask for valid positions
    channel_weights : list/tuple of 4 floats or None
        Weights for each channel [dist, omega, theta, phi].
        If None, uses equal weights [0.25, 0.25, 0.25, 0.25].
        Example: [0.1, 0.3, 0.3, 0.3] gives more weight to angles.

    Returns
    -------
    loss : total weighted loss
    rec : reconstruction loss
    kld : KL divergence loss
    """
    # Set default channel weights if not provided
    if channel_weights is None:
        channel_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights

    # Convert to tensor for computation
    channel_weights = torch.tensor(channel_weights, device=x.device, dtype=x.dtype)
    # Reshape to [1, C, 1, 1] for broadcasting
    channel_weights = channel_weights.view(1, -1, 1, 1)

    if mask is None:
        # Compute per-channel MSE: [B, C, H, W]
        mse_per_channel = (recon - x).pow(2)
        # Apply channel weights
        weighted_mse = mse_per_channel * channel_weights
        # Average over all dimensions
        rec = weighted_mse.mean()
    else:
        # recon, x: [B, C, H, W]; mask: [B, H, W]
        mse_per_channel = (recon - x).pow(2)  # [B, C, H, W]
        # Apply channel weights
        weighted_mse = mse_per_channel * channel_weights  # [B, C, H, W]
        # Expand mask to match channels: [B, 1, H, W]
        mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
        # Zero-out invalid positions
        weighted_mse = weighted_mse * mask_expanded  # [B, C, H, W]
        # Sum over all valid elements and normalize
        rec = weighted_mse.sum() / mask.sum().clamp_min(1.0)

    # KL divergence (unchanged)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = rec_w * rec + kl_w * kld

    return loss, rec, kld
