# Training loop logic. to be implemented in main.py
import torch

def train_vanilla_vae(
    vae,
    train_loader,
    vae_loss_fn,          
    epochs,
    device,
    lr,
    rec_w,
    kl_w,
    log_every,
    betas=(0.9, 0.999)
):
    vae.to(device) 
    vae.train()

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)  

    step = 0
    for ep in range(epochs):
        for batch in train_loader:
            # batch can be x or (x, mask)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, mask = batch[0], batch[1]
            else:
                x, mask = batch, None

            x = x.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            # Forward
            recon, mu, logvar = vae(x)

            # Loss (masked or unmasked, depending on mask)
            loss, rec, kld = vae_loss_fn(
                recon, x, mu, logvar, mask=mask, rec_w=rec_w, kl_w=kl_w
            )

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(
                    f"ep {ep:03d} step {step:06d} | "
                    f"loss {loss.item():.4f} | rec {rec.item():.4f} | kld {kld.item():.4f}"
                )
            step += 1

    return vae
