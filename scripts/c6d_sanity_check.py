import torch
def parse_arguments():
    import argparse as ap
    parser = ap.ArgumentParser(description="Check c6d and mask tensors for NaN/Inf values")
    parser.add_argument(
        "--input_c6d",
        type=str,
        required=True,
        help="Path to input .pt file containing c6d and mask tensors"
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()
    input_c6d = args.input_c6d
    d = torch.load(input_c6d, map_location="cpu")
    c6d = d["c6d"]
    mask = d["mask"]

    # Check for NaN/Inf
    bad_c6d = ~torch.isfinite(c6d).all(dim=(1,2,3))
    bad_mask = ~torch.isfinite(mask).all(dim=(1,2))
    bad_indices = bad_c6d | bad_mask

    print("bad c6d count:", bad_c6d.sum().item())
    print("bad mask count:", bad_mask.sum().item())
    print("total bad:", bad_indices.sum().item())

    if bad_indices.sum().item() > 0:
        # Remove bad samples
        keep = ~bad_indices
        d["c6d"] = c6d[keep]
        d["mask"] = mask[keep]
        
        # Save
        torch.save(d, input_c6d)
        print("saved - new N =", d["c6d"].shape[0])
    else:
        print("N = ", d["c6d"].shape[0])


# # Single multi-model PDB file
# python Prep_PDB42DTF.py --pdb_file all_structures.pdb --max_res 40

# # Multiple separate PDB files (original behavior)
# python Prep_PDB42DTF.py --pdb_folder ./pdb_files/ --max_res 40
