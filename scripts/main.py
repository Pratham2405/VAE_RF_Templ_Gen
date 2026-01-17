# main.py

import os
import argparse as ap
import torch
from VAE_model import VanillaVAE
from vae_loss import vae_loss
from dataloader import build_train_loader
from train import train_vanilla_vae

def parse_args():
    parser = ap.ArgumentParser(description="2D template VAE training")

    parser.add_argument(
        "--protein_length",
        type=int,
        required=True,
        help="Length of the target protein(number of residues).",
    )

    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to the .pt file containing c6d and mask tensors",
    )

    parser.add_argument(
        "--kl_div_weight",
        type=float,
        default=1e-3,
        help="Weight for KL divergence term in the VAE loss",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=42,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )

    parser.add_argument(
        "--output_weights",
        type=str,
        default="vae_weights.pth",
        help="Path to save trained VAE weights",
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Loss printing frequency",
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=64,
        help="Feature Dimension",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Latent Dimension",
    )

    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=True,
        help="Pin Memory",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Num Workers",
    )

    parser.add_argument(
        "--channel_weights",
        type=float,
        nargs=4,
        default=None,
        metavar=('W_DIST', 'W_OMEGA', 'W_THETA', 'W_PHI'),
        help="Per-channel loss weights for [distance, omega, theta, phi]. "
             "Example: --channel_weights 0.1 0.3 0.3 0.3 to emphasize angles. "
             "Default: None (equal weights [0.25, 0.25, 0.25, 0.25])",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    training_data = args.training_data
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    kl_w = args.kl_div_weight
    output_weights = args.output_weights
    protein_length = args.protein_length
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    latent_dim = args.latent_dim
    feature_dim = args.feature_dim
    log_every = args.log_every
    channel_weights = args.channel_weights  

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Protein length: {protein_length}")
    print(f"Training data: {training_data}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL weight: {kl_w}")
    print(f"Feature dim: {feature_dim}")
    print(f"Latent dim: {latent_dim}")
    if channel_weights is not None:
        print(f"Channel weights: {channel_weights}")
        print(f"  Distance: {channel_weights[0]:.3f}")
        print(f"  Omega:    {channel_weights[1]:.3f}")
        print(f"  Theta:    {channel_weights[2]:.3f}")
        print(f"  Phi:      {channel_weights[3]:.3f}")
    else:
        print(f"Channel weights: Equal (0.25 each)")
    print("=" * 60)

    # DataLoader built from the .pt data (c6d + mask)
    train_loader = build_train_loader(training_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Model: 4 channels (dist, omega, theta, phi)
    vae = VanillaVAE(feature_dim=feature_dim, input_size=protein_length, latent_dim=latent_dim, input_channels=4)

    # Train
    vae = train_vanilla_vae(
        vae,
        train_loader,
        vae_loss_fn=vae_loss,
        epochs=epochs,
        device=device,
        lr=learning_rate,
        rec_w=1.0,
        kl_w=kl_w,
        log_every=log_every,
        channel_weights=channel_weights,  
    )

    # Save ckpt and weights
    torch.save(vae.state_dict(), output_weights)
    abs_output_weights = os.path.realpath(output_weights)
    print(f"Model weights saved to {abs_output_weights}")

    ckpt = {
        "config": {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "protein_length": protein_length,
            "input_channels": 4,
            "output_weights": output_weights,
            "channel_weights": channel_weights,  
        },
        "epochs": epochs,
    }

    torch.save(ckpt, "vae_ckpt.pt")
    print("Checkpoint saved to vae_ckpt.pt")
