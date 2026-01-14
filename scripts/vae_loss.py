# losses/vae_loss.py
import torch
import torch.nn.functional as F

def vae_loss(recon, x, mu, logvar, rec_w, kl_w, mask=None):
    if mask is None:
        # simple unmasked MSE
        rec = F.mse_loss(recon, x, reduction="mean")
    else:
        # recon, x: [B, C, H, W]; mask: [B, H, W]
        mse = (recon - x).pow(2)          # [B, C, H, W]
        mse = mse.mean(dim=1)             # average over channels -> [B, H, W]
        mse = mse * mask                  # zero-out invalid positions
        rec = mse.sum() / mask.sum().clamp_min(1.0)

    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec_w * rec + kl_w * kld
    return loss, rec, kld
