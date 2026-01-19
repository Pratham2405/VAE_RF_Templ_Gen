import torch
import argparse as ap
import numpy as np

def parse_arguments():
    parser = ap.ArgumentParser(description="Check and clean c6d data for invalid values")
    parser.add_argument(
        "--input_c6d",
        type=str,
        required=True,
        help="Path to .pt file containing c6d and mask tensors (will be overwritten if bad samples found)"
    )
    parser.add_argument(
        "--distance_max",
        type=float,
        default=25.0,
        help="Maximum valid distance value (default: 25.0 Å)"
    )
    return parser.parse_args()

def check_and_clean(c6d, mask, dist_max=25.0):
    """Check c6d tensor for invalid values and return indices of bad samples"""
    bad_indices = set()
    N = c6d.shape[0]

    print("\n" + "="*60)
    print("C6D SANITY CHECK")
    print("="*60)
    print(f"Total samples: {N}")
    print(f"Shape: c6d={c6d.shape}, mask={mask.shape}")

    # 1. NaN check
    print("\n[1/5] Checking NaN values...")
    nan_c6d = torch.isnan(c6d).any(dim=(1,2,3))
    nan_mask = torch.isnan(mask).any(dim=(1,2))
    nan_idx = torch.where(nan_c6d | nan_mask)[0].tolist()
    if nan_idx:
        print(f"  ❌ {len(nan_idx)} samples with NaN")
        bad_indices.update(nan_idx)
    else:
        print("  ✓ No NaN")

    # 2. Inf check
    print("\n[2/5] Checking Inf values...")
    inf_c6d = torch.isinf(c6d).any(dim=(1,2,3))
    inf_mask = torch.isinf(mask).any(dim=(1,2))
    inf_idx = torch.where(inf_c6d | inf_mask)[0].tolist()
    if inf_idx:
        print(f"  ❌ {len(inf_idx)} samples with Inf")
        bad_indices.update(inf_idx)
    else:
        print("  ✓ No Inf")

    # 3. Distance validation
    print("\n[3/5] Checking distances...")
    dist = c6d[..., 0]
    valid_mask = dist < 999.0  # Exclude sentinel values
    invalid_dist = ((dist < 0) | (dist > dist_max)) & valid_mask
    dist_idx = torch.where(invalid_dist.any(dim=(1,2)))[0].tolist()
    if dist_idx:
        print(f"  ❌ {len(dist_idx)} samples with invalid distances")
        print(f"     Valid range: [0, {dist_max}] Å")
        bad_indices.update(dist_idx)
    else:
        print(f"  ✓ All distances in [0, {dist_max}] Å")

    # Show distance statistics
    valid_dist = dist[valid_mask]
    if len(valid_dist) > 0:
        print(f"     Stats: min={valid_dist.min():.2f}, max={valid_dist.max():.2f}, "
              f"mean={valid_dist.mean():.2f}, std={valid_dist.std():.2f}")

    # 4. Angle validation
    print("\n[4/5] Checking angles...")
    angle_names = ["omega (dihedral)", "theta (dihedral)", "phi (planar)"]
    angle_ranges = [(-np.pi, np.pi), (-np.pi, np.pi), (0, np.pi)]

    for ch, (name, (vmin, vmax)) in enumerate(zip(angle_names, angle_ranges), 1):
        angles = c6d[..., ch]
        tolerance = 0.1  # Small tolerance for floating point errors
        invalid = (angles < vmin - tolerance) | (angles > vmax + tolerance)
        ang_idx = torch.where(invalid.any(dim=(1,2)))[0].tolist()

        if ang_idx:
            print(f"  ❌ Ch{ch} ({name}): {len(ang_idx)} samples out of range")
            print(f"     Valid range: [{vmin:.3f}, {vmax:.3f}]")
            bad_indices.update(ang_idx)
        else:
            print(f"  ✓ Ch{ch} ({name}): all in [{vmin:.3f}, {vmax:.3f}]")

        # Show angle statistics
        print(f"     Stats: min={angles.min():.3f}, max={angles.max():.3f}, "
              f"mean={angles.mean():.3f}, std={angles.std():.3f}")

    # 5. Mask validation
    print("\n[5/5] Checking mask...")
    if mask.dtype == torch.bool:
        print("  ✓ Boolean mask")
    elif ((mask < 0) | (mask > 1)).any():
        print(f"  ❌ Invalid mask values (should be 0/1 or boolean)")
        print(f"     Range: min={mask.min()}, max={mask.max()}")
    else:
        print("  ✓ Valid mask values (0 or 1)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {N}")
    print(f"Bad samples: {len(bad_indices)}")
    print(f"Clean samples: {N - len(bad_indices)}")

    if bad_indices:
        print(f"\nBad sample indices (first 20):")
        print(f"  {sorted(list(bad_indices))[:20]}")

    return bad_indices

if __name__ == "__main__":
    args = parse_arguments()

    print(f"Loading {args.input_c6d}...")
    data = torch.load(args.input_c6d, map_location='cpu')
    c6d = data["c6d"]
    mask = data["mask"]

    # Check for bad samples
    bad_indices = check_and_clean(c6d, mask, dist_max=args.distance_max)

    # Clean and save if needed
    if len(bad_indices) > 0:
        print(f"\n{'='*60}")
        print("CLEANING DATA")
        print("="*60)
        keep = [i for i in range(len(c6d)) if i not in bad_indices]
        cleaned_c6d = c6d[keep]
        cleaned_mask = mask[keep]

        torch.save({"c6d": cleaned_c6d, "mask": cleaned_mask}, args.input_c6d)
        print(f"✓ Removed {len(bad_indices)} bad samples")
        print(f"✓ Saved {len(keep)} clean samples to {args.input_c6d}")
        print(f"✓ New c6d shape: {cleaned_c6d.shape}")
        print("="*60)
    else:
        print(f"\n{'='*60}")
        print("✓ NO ISSUES FOUND - File unchanged")
        print("="*60)

# Usage:
# python c6d_sanity_check.py --input_c6d train_c6d.pt
# python c6d_sanity_check.py --input_c6d test_c6d.pt
# python c6d_sanity_check.py --input_c6d train_c6d.pt --distance_max 30.0
