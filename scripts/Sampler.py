import torch
import numpy as np
import argparse as ap
from VAE_model import VanillaVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Sampler():
    def __init__(self, vae_model, latent_dim, device, num_samples):
        self.vae = vae_model
        self.latent_dim = latent_dim
        self.device = device
        self.num_samples = num_samples

    def sample_prior(self, num_samples):
        """Generate samples from the prior distribution (random latent vectors)"""
        vae = self.vae
        samples = []
        latent_dim = self.latent_dim

        # Batch sampling for efficiency
        batch_size = min(32, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                z = torch.randn(current_batch_size, latent_dim, device=self.device)
                decoded = vae.decode(z)  # Shape: [batch_size, 4, L, L]

                for j in range(current_batch_size):
                    samples.append(decoded[j].cpu().numpy())

        return samples

    def sample_posterior(self, batch, num_samples_per_input=1):
        """Generate samples from the posterior distribution given input data

        Args:
            batch: Input tensor of shape [B, 4, L, L]
            num_samples_per_input: Number of samples to generate per input
        """
        vae = self.vae
        samples = []
        originals = []

        with torch.no_grad():
            # Process each item in the batch
            for i in range(batch.shape[0]):
                x = batch[i:i+1]  # Keep batch dimension: [1, 4, L, L]

                # Generate multiple samples from this input
                for _ in range(num_samples_per_input):
                    mu, logvar = vae.encode(x)
                    z = vae.reparameterize(mu, logvar)
                    reconstructed = vae.decode(z)  # [1, 4, L, L]
                    samples.append(reconstructed.squeeze(0).cpu().numpy())

                # Store original for comparison (only once per input)
                originals.append(x.squeeze(0).cpu().numpy())

        return samples, originals

    def calculate_reconstruction_loss(self, generated, reference, mask=None, mode='posterior'):
        """Calculate MSE reconstruction loss between generated and reference data

        Args:
            generated: Tensor [N, 4, L, L] - generated/reconstructed samples
            reference: Tensor [N, 4, L, L] for posterior or [M, 4, L, L] for prior
            mask: Optional mask tensor [N, L, L] or [M, L, L]
            mode: 'posterior' (1-to-1 matching) or 'prior' (find closest match)

        Returns:
            dict with loss statistics
        """
        # Convert to tensors if needed
        if isinstance(generated, list):
            generated = torch.tensor(np.stack(generated), device=self.device)
        elif not isinstance(generated, torch.Tensor):
            generated = torch.tensor(generated, device=self.device)
        else:
            generated = generated.to(self.device)

        if isinstance(reference, list):
            reference = torch.tensor(np.stack(reference), device=self.device)
        elif not isinstance(reference, torch.Tensor):
            reference = torch.tensor(reference, device=self.device)
        else:
            reference = reference.to(self.device)

        losses = []

        if mode == 'posterior':
            # Posterior: 1-to-1 comparison (each generated vs its original)
            assert generated.shape[0] == reference.shape[0], \
                f"Shape mismatch in posterior mode: generated {generated.shape[0]} vs reference {reference.shape[0]}"

            for i in range(generated.shape[0]):
                if mask is not None:
                    # Get mask for this specific sample and expand to all channels
                    mask_i = mask[i].to(self.device)  # [L, L]
                    mask_expanded = mask_i.unsqueeze(0).expand(4, -1, -1)  # [4, L, L]

                    # Apply mask and calculate normalized loss
                    masked_gen = generated[i] * mask_expanded
                    masked_ref = reference[i] * mask_expanded

                    # Calculate MSE only over masked (valid) elements
                    num_valid = mask_expanded.sum().clamp(min=1.0)
                    loss = ((masked_gen - masked_ref) ** 2).sum() / num_valid
                else:
                    loss = F.mse_loss(generated[i], reference[i])

                losses.append(loss.item())

        else:  # mode == 'prior'
            # Prior: find closest match in reference set for each generated sample
            for i in range(generated.shape[0]):
                sample_losses = []

                for j in range(reference.shape[0]):
                    if mask is not None:
                        # Get mask for reference sample j
                        mask_j = mask[j].to(self.device)  # [L, L]
                        mask_expanded = mask_j.unsqueeze(0).expand(4, -1, -1)  # [4, L, L]

                        # Apply mask
                        masked_gen = generated[i] * mask_expanded
                        masked_ref = reference[j] * mask_expanded

                        # Calculate normalized MSE
                        num_valid = mask_expanded.sum().clamp(min=1.0)
                        loss = ((masked_gen - masked_ref) ** 2).sum() / num_valid
                    else:
                        loss = F.mse_loss(generated[i], reference[j])

                    sample_losses.append(loss.item())

                # Use minimum loss (closest match)
                losses.append(min(sample_losses))

        loss_array = np.array(losses)

        return {
            'mean_loss': float(loss_array.mean()),
            'std_loss': float(loss_array.std()),
            'min_loss': float(loss_array.min()),
            'max_loss': float(loss_array.max()),
            'median_loss': float(np.median(loss_array))
        }

    def plot_histogram(self, generated, reference, mask=None, sampling_type="prior", 
                      save_path="histogram_comparison.png"):
        """Plot histogram comparison for each of the 4 channels

        Args:
            generated: numpy array or tensor [N, 4, L, L]
            reference: numpy array or tensor [M, 4, L, L]
            mask: Optional mask tensor [M, L, L] to filter out invalid values
            sampling_type: "prior" or "posterior" for labeling
            save_path: Path to save the figure
        """
        # Convert to numpy if needed
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        elif isinstance(generated, list):
            generated = np.stack(generated)

        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()

        # Create mask for filtering (if provided)
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            # Expand mask to match channels: [M, L, L] -> [M, 4, L, L]
            mask_expanded = np.expand_dims(mask, axis=1)  # [M, 1, L, L]
            mask_expanded = np.repeat(mask_expanded, 4, axis=1)  # [M, 4, L, L]
        else:
            mask_expanded = None

        # Create figure with 4 subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        channel_names = ['Channel 0 (ω)', 'Channel 1 (θ)', 'Channel 2 (φ)', 'Channel 3 (dist)']

        for c in range(4):
            ax = axes[c]

            # Get channel data
            gen_chan = generated[:, c, :, :]
            ref_chan = reference[:, c, :, :]

            # Apply mask if provided
            if mask_expanded is not None:
                # For reference data
                mask_chan = mask_expanded[:, c, :, :].astype(bool)
                ref_chan = ref_chan[mask_chan]

                # For generated data - create appropriate mask
                # If shapes match, use same mask; otherwise skip masking for generated
                if generated.shape[0] == reference.shape[0]:
                    gen_mask = mask_expanded[:, c, :, :].astype(bool)
                    gen_chan = gen_chan[gen_mask]
                else:
                    gen_chan = gen_chan.ravel()
            else:
                # Flatten all values
                gen_chan = gen_chan.ravel()
                ref_chan = ref_chan.ravel()

            # Filter out sentinel/invalid values (e.g., values near 999 or 1000)
            valid_mask_gen = (gen_chan < 900) & (gen_chan > -900)
            valid_mask_ref = (ref_chan < 900) & (ref_chan > -900)

            gen_chan = gen_chan[valid_mask_gen]
            ref_chan = ref_chan[valid_mask_ref]

            # Plot histograms
            ax.hist(ref_chan, bins=60, alpha=0.6, color="blue",
                   label=f"Reference ({'Training' if sampling_type=='prior' else 'Test'} Data)",
                   density=True)
            ax.hist(gen_chan, bins=60, alpha=0.6, color="red",
                   label="Generated Samples", density=True)

            ax.set_title(channel_names[c], fontsize=12, fontweight='bold')
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Distribution Comparison - {sampling_type.capitalize()} Sampling',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
        plt.close()

        return fig

def sample_args():
    parser = ap.ArgumentParser(
        description="Sampling script for VAE. Prior sampling generates new samples from random latent vectors. "
                   "Posterior sampling reconstructs test data through the VAE encoder-decoder."
    )

    parser.add_argument(
        "--sampling_type",
        type=str,
        required=True,
        choices=["prior", "posterior"],
        help="Choose between 'posterior' and 'prior' sampling."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (for prior) or samples per input (for posterior)."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_c6d_samples.pt",
        help="Path to save the generated samples."
    )

    parser.add_argument(
        "--test_batch",
        type=str,
        default=None,
        help="Path to the test data file (required for posterior sampling). Expected format: dict with 'c6d' and 'mask' keys."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (cpu or cuda)."
    )

    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to the training data file for comparison. Expected format: dict with 'c6d' and 'mask' keys."
    )

    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="vae_ckpt.pt",
        help="Path to the VAE checkpoint file containing config."
    )

    parser.add_argument(
        "--vae_weights",
        type=str,
        required=True,
        help="Path to the trained VAE model weights."
    )

    parser.add_argument(
        "--histogram_output",
        type=str,
        required=True,
        help="Path to save histogram comparison plot."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = sample_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Validate arguments
    sampling_type = args.sampling_type.lower()
    if sampling_type == "posterior" and args.test_batch is None:
        raise ValueError("--test_batch is required for posterior sampling.")

    # Model configuration
    latent_dim = 256
    feature_dim = 64
    protein_length = 40

    # Load training data
    print(f"Loading training data from {args.training_data}...")
    training_data_dict = torch.load(args.training_data, map_location=device)
    training_c6d = training_data_dict['c6d']  # Shape: [N, 4, L, L] or [N, L, L, 4]
    training_mask = training_data_dict.get('mask', None)

    # Ensure correct channel order: [N, 4, L, L]
    if training_c6d.shape[1] != 4:
        training_c6d = training_c6d.permute(0, 3, 1, 2).contiguous()

    print(f"Training data shape: {training_c6d.shape}")

    # Initialize VAE model
    print("Initializing VAE model...")
    vae_model = VanillaVAE(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        input_size=protein_length,
        input_channels=4
    )

    vae_model.load_state_dict(torch.load(args.vae_weights, map_location=device))
    vae_model.to(device)
    vae_model.eval()
    print("VAE model loaded successfully.")

    # Initialize sampler
    sampler = Sampler(
        vae_model=vae_model,
        latent_dim=latent_dim,
        device=device,
        num_samples=args.num_samples
    )

    # Perform sampling
    if sampling_type == "prior":
        print(f"\nPerforming prior sampling ({args.num_samples} samples)...")
        generated_samples = sampler.sample_prior(num_samples=args.num_samples)
        generated_tensor = torch.tensor(np.stack(generated_samples))
        reference_data = training_c6d
        reference_mask = training_mask
        loss_mode = 'prior'
        print(f"Generated {len(generated_samples)} samples from prior distribution.")

    else:  # posterior
        print(f"\nPerforming posterior sampling...")
        test_data_dict = torch.load(args.test_batch, map_location=device)
        test_c6d = test_data_dict['c6d']
        test_mask = test_data_dict.get('mask', None)

        # Ensure correct channel order: [B, 4, L, L]
        if test_c6d.shape[1] != 4:
            test_c6d = test_c6d.permute(0, 3, 1, 2).contiguous()

        print(f"Test data shape: {test_c6d.shape}")

        # Select subset if needed
        if test_c6d.shape[0] > args.num_samples:
            indices = torch.randperm(test_c6d.shape[0])[:args.num_samples]
            print(f"Randomly selecting {args.num_samples} from {test_c6d.shape[0]} samples")
            test_c6d = test_c6d[indices]
            if test_mask is not None:
                test_mask = test_mask[indices]

        # Generate reconstructions
        generated_samples, originals = sampler.sample_posterior(
            batch=test_c6d,
            num_samples_per_input=1  # One reconstruction per input
        )

        generated_tensor = torch.tensor(np.stack(generated_samples))
        # CRITICAL: Compare reconstructions to their ORIGINALS, not to test_c6d
        reference_data = torch.tensor(np.stack(originals), device=device)
        reference_mask = test_mask
        loss_mode = 'posterior'

        print(f"Generated {len(generated_samples)} reconstructed samples.")

    # Save generated samples
    torch.save(generated_tensor, args.output_file)
    print(f"\nGenerated samples saved to {args.output_file}")
    print(f"Output shape: {generated_tensor.shape}")

    # Calculate reconstruction loss
    print("\nCalculating reconstruction loss...")
    loss_stats = sampler.calculate_reconstruction_loss(
        generated=generated_tensor,
        reference=reference_data,
        mask=reference_mask,
        mode=loss_mode
    )

    print("\nReconstruction Loss Statistics:")
    print(f"  Mean Loss: {loss_stats['mean_loss']:.6f}")
    print(f"  Std Loss:  {loss_stats['std_loss']:.6f}")
    print(f"  Min Loss:  {loss_stats['min_loss']:.6f}")
    print(f"  Max Loss:  {loss_stats['max_loss']:.6f}")
    print(f"  Median Loss: {loss_stats['median_loss']:.6f}")

    # Plot histograms
    print("\nGenerating histogram comparison...")
    sampler.plot_histogram(
        generated=generated_tensor,
        reference=reference_data,
        mask=reference_mask,
        sampling_type=sampling_type,
        save_path=args.histogram_output
    )

    print("\n" + "="*60)
    print("Sampling and evaluation complete!")
    print(f"  Generated samples: {args.output_file}")
    print(f"  Histogram plot: {args.histogram_output}")
    print("="*60)
