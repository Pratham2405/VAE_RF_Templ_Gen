# import torch
# import numpy as np
# import argparse as ap
# from VAE_model import VanillaVAE
# from main import parse_args
# import matplotlib.pyplot as plt
# from Prep_PDB42DTF import 

# class Sampler():

#     def __init__(self, vae_model, latent_dim, device, num_samples):
#         self.vae = vae_model
#         self.latent_dim = latent_dim
#         self.device = device
#         self.num_samples = num_samples
       

#     def sample_prior(self, num_samples):
#         vae = self.vae
#         samples = []
#         latent_dim = self.latent_dim
#         with torch.no_grad():
#             for _ in range(num_samples):
#                 z = torch.randn(1, latent_dim, device=self.device)
#                 z = vae.decode(z).squeeze().cpu().numpy()
#                 samples.append(z)
#         return samples
    
#     def sample_posterior(self, batch, num_samples=8):
#         vae = self.vae
#         samples = []
#         with torch.no_grad():
#             for _ in range(num_samples):
#                 mu, logvar = vae.encode(batch)
#                 z = vae.reparameterize(mu, logvar)
#                 x = vae.decode(z).squeeze().cpu().numpy()            
#                 samples.append(x)
#         return samples

# def sample_args(prior=True):
#     parser = ap.ArgumentParser(description="Provide arguments for the sampling script. Prior sampling requires the training data c6d path and the goal template you want to generate; Posterior sampling requires you to provide the test data points(path) you want to parse through the VAE for evaluation purposes.")
#     parser.add_argument(
#         "--sampling_type",
#         type=str,
#         required=True,
#         help="Choose between posterior and prior sampling."
#     )
#     parser.add_argument(
#         "--num-samples",
#         type=int,
#         default=10,
#         help="Number of pairwise feature files to generate."
#     )
#     parser.add_argument(
#         "--output-file",
#         type=str,
#         default="generated_c6d_samples.pt",
#         help="Path to the output samples."
#     )
#     parser.add_argument(
#         "--test_batch",
#         type=str,
#         required=True if not prior else False,
#         help="Path to the test samples for prior sampling."
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cpu",
#         help="Device for loading the batch data set"
#     )
#     parser.add_argument(
#         "--training_data",
#         type=str,
#         default="all_c6d_with_mask.pt",
#         help= "The training data consolidated c6d file for comparison"
#     )
#     return parser.parse_args()

# def plot_histogram(x, y):
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     axes = axes.ravel()

#     for c in range(4):
#         ax = axes[c]
#         x_chan = x[:, c, :, :].cpu().numpy().ravel()
#         y_chan = y[:, c, :, :].cpu().numpy().ravel()

#         ax.hist(x_chan, bins=50, alpha=0.5, color="blue", label="X")
#         ax.hist(y_chan, bins=50, alpha=0.5, color="red", label="Y")
#         ax.set_title(f"Channel {c}")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Count")
#         ax.legend()

#     plt.tight_layout()
#     plt.show()

#     return fig

# def plot_ramachandran():
#     pass

      

# if __name__=="__main__":
#     args = sample_args()
#     device = args.device
#     ckpt = torch.load("vae_ckpt.pt", map_location=device)
#     cfg = ckpt["config"]
#     num_samples = args.num_samples
#     output_samples = args.output_file
    
#     sampling_type = args.sampling_type
#     latent_dim = cfg["latent_dim"]
#     feature_dim = cfg["feature_dim"]
#     protein_length = cfg["protein_length"]
#     output_weights = cfg["output_weights"]
#     test_batch = args.test_batch
#     training_data = args.training_data

#     prepared_batch = 
#     if sampling_type.lower() not in ["prior", "posterior"]:
#         raise ValueError("--sampling_type must be either 'prior' or 'posterior'")
#     elif sampling_type.lower() == "posterior" and test_batch == None:
#         raise ValueError("--test_batch cannot be None when choosing posterior sampling.")

#     vae_model = VanillaVAE(latent_dim=latent_dim, feature_dim=feature_dim, input_size=protein_length, input_channels=4)
 
#     vae_model.load_state_dict(torch.load(output_weights, map_location=device))
#     vae_model.to(device)
#     vae_model.eval()
#     training_data = torch.load(training_data, map_location=device)
#     batch = torch.load(test_batch, map_location=device)
#     sampler = Sampler(vae_model, latent_dim=latent_dim, device=device, num_samples=num_samples)
#     if sampling_type.lower()=="prior":
#         output = sampler.sample_prior(num_samples=num_samples)
        
#     else:
#         output = sampler.sample_posterior(batch=batch, num_samples=num_samples)
    
#     output_tensor = torch.tensor(np.stack(output))
#     torch.save(output_tensor, output_samples)

# # prior sampling --> loss minimisation + reconstruction loss calculation + histogram plotting + ramachandran plotting.
# # posterior sampling --> reconstruction loss calculation + histogram plotting + ramachandran plotting.
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

                # Store original for comparison
                originals.append(x.squeeze(0).cpu().numpy())

        return samples, originals

    def calculate_reconstruction_loss(self, generated, reference, mask=None):
        """Calculate MSE reconstruction loss between generated and reference data

        Args:
            generated: List of numpy arrays or tensor [N, 4, L, L]
            reference: Tensor [M, 4, L, L] (training or test data)
            mask: Optional mask tensor [M, L, L]

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

        if not isinstance(reference, torch.Tensor):
            reference = torch.tensor(reference, device=self.device)
        else:
            reference = reference.to(self.device)

        # Calculate pairwise MSE between all generated and reference samples
        losses = []
        for gen_sample in generated:
            sample_losses = []
            for ref_sample in reference:
                if mask is not None:
                    # Apply mask if provided
                    # mask_expanded = mask.unsqueeze(0).expand(4, -1, -1)
                    mask_expanded = mask.unsqueeze(1).expand(-1, 4, -1, -1)
                    loss = F.mse_loss(gen_sample * mask_expanded, ref_sample * mask_expanded)
                else:
                    loss = F.mse_loss(gen_sample, ref_sample)
                sample_losses.append(loss.item())

            # Use minimum loss (closest match in reference set)
            losses.append(min(sample_losses))

        loss_array = np.array(losses)
        return {
            'mean_loss': float(loss_array.mean()),
            'std_loss': float(loss_array.std()),
            'min_loss': float(loss_array.min()),
            'max_loss': float(loss_array.max()),
            'median_loss': float(np.median(loss_array))
        }

    def plot_histogram(self, generated, reference, sampling_type="prior", save_path="histogram_comparison.png"):
        """Plot histogram comparison for each of the 4 channels

        Args:
            generated: numpy array or tensor [N, 4, L, L]
            reference: numpy array or tensor [M, 4, L, L]
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

        # Create figure with 4 subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        channel_names = ['Channel 0 (ω)', 'Channel 1 (θ)', 'Channel 2 (φ)', 'Channel 3 (dist)']

        for c in range(4):
            ax = axes[c]

            # Flatten all values for this channel
            gen_chan = generated[:, c, :, :].ravel()
            ref_chan = reference[:, c, :, :].ravel()

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

    # def plot_ramachandran(self, generated, reference, sampling_type="prior", save_path="ramachandran_comparison.png"):
    #     """Plot Ramachandran-style plots for dihedral angles

    #     Assuming channels represent angular features that can be visualized in phi-psi space.
    #     This plots channel 1 vs channel 2 as a proxy for backbone dihedral angles.
    #     """
    #     # Convert to numpy if needed
    #     if isinstance(generated, torch.Tensor):
    #         generated = generated.cpu().numpy()
    #     elif isinstance(generated, list):
    #         generated = np.stack(generated)

    #     if isinstance(reference, torch.Tensor):
    #         reference = reference.cpu().numpy()

    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Extract angular channels (assuming channels 1 and 2 represent angles)
        # For 2D pairwise features, we sample diagonal and near-diagonal elements
        # def extract_angles(data):
        #     """Extract representative angle pairs from the pairwise feature tensor"""
        #     phi_vals = []
        #     psi_vals = []
        #     for sample in data:
        #         # Sample from diagonal and off-diagonal elements
        #         for i in range(0, sample.shape[2] - 1):
        #             phi_vals.append(sample[1, i, i+1])  # Channel 1
        #             psi_vals.append(sample[2, i, i+1])  # Channel 2
        #     return np.array(phi_vals), np.array(psi_vals)

        # ref_phi, ref_psi = extract_angles(reference)
        # gen_phi, gen_psi = extract_angles(generated)

        # # Plot reference data
        # axes[0].hexbin(ref_phi, ref_psi, gridsize=50, cmap='Blues', mincnt=1)
        # axes[0].set_title(f"Reference ({'Training' if sampling_type=='prior' else 'Test'} Data)", 
        #                  fontsize=12, fontweight='bold')
        # axes[0].set_xlabel('φ (Channel 1)', fontsize=10)
        # axes[0].set_ylabel('ψ (Channel 2)', fontsize=10)
        # axes[0].grid(True, alpha=0.3)

        # # Plot generated data
        # axes[1].hexbin(gen_phi, gen_psi, gridsize=50, cmap='Reds', mincnt=1)
        # axes[1].set_title("Generated Samples", fontsize=12, fontweight='bold')
        # axes[1].set_xlabel('φ (Channel 1)', fontsize=10)
        # axes[1].set_ylabel('ψ (Channel 2)', fontsize=10)
        # axes[1].grid(True, alpha=0.3)

        # plt.suptitle(f'Ramachandran-Style Plot - {sampling_type.capitalize()} Sampling', 
        #             fontsize=14, fontweight='bold')
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print(f"Ramachandran plot saved to {save_path}")
        # plt.close()

        # return fig


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

    # Load checkpoint and configuration
    # print(f"Loading checkpoint from {args.vae_checkpoint}...")
    # ckpt = torch.load(args.vae_checkpoint, map_location=device)
    # cfg = ckpt["config"]

    # latent_dim = cfg["latent_dim"]
    # feature_dim = cfg["feature_dim"]
    # protein_length = cfg["protein_length"]
    latent_dim = 256
    feature_dim = 64
    protein_length = 40

    # Load training data
    print(f"Loading training data from {args.training_data}...")
    training_data_dict = torch.load(args.training_data, map_location=device)
    training_c6d = training_data_dict['c6d']  # Shape: [N, 4, L, L]
    training_mask = training_data_dict.get('mask', None)
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

        print(f"Generated {len(generated_samples)} samples from prior distribution.")

    else:  # posterior
        print(f"\nPerforming posterior sampling...")
        test_data_dict = torch.load(args.test_batch, map_location=device)
        test_c6d = test_data_dict['c6d']  # Shape: [B, 4, L, L]
        test_mask = test_data_dict.get('mask', None)
        test_c6d = test_c6d.permute(0, 3, 1, 2).contiguous()

        print(f"Test data shape: {test_c6d.shape}")

        if test_c6d.shape[0] > args.num_samples:
            indices = torch.randperm(test_c6d.shape[0])[:args.num_samples]
            print(f"Randomly selecting {args.num_samples} from {test_c6d.shape[0]} samples")
            test_c6d = test_c6d[indices]
            if test_mask is not None:
                test_mask = test_mask[indices]
            
        generated_samples, originals = sampler.sample_posterior(
            batch=test_c6d,
            num_samples_per_input=max(1, args.num_samples // test_c6d.shape[0])
        )
        generated_tensor = torch.tensor(np.stack(generated_samples))
        reference_data = test_c6d
        reference_mask = test_mask

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
        mask=reference_mask
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
        sampling_type=sampling_type,
        save_path=args.histogram_output
    )

    print("\n" + "="*60)
    print("Sampling and evaluation complete!")
    print(f"  Generated samples: {args.output_file}")
    print(f"  Histogram plot: {args.histogram_output}")
    print("="*60)

    # Example usage:
    # python Sampler.py --sampling_type prior --num_samples 100 --output_file prior_samples.pt --training_data all_c6d_with_mask.pt --vae_weights vae_weights.pth --histogram_output prior_histogram.png
    # python Sampler.py --sampling_type posterior --num_samples 50 --output_file posterior_samples.pt --test_batch test_c6d_with_mask.pt --training_data all_c6d_with_mask.pt --vae_weights vae_weights.pth --histogram_output posterior_histogram.png