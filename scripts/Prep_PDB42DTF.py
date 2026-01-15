# import os
# import torch
# import argparse as ap

# def parse_pdb_backbone(pdb_path, max_res):
#     arr = torch.zeros(max_res, 3, 3)
#     with open(pdb_path, 'r') as f:
#         for line in f:
#             if line.startswith("ATOM"):
#                 atom_name = line[12:16].strip()
#                 res_seq = int(line[22:26].strip())
#                 if 1 <= res_seq <= max_res and atom_name in ["N", "CA", "C"]:
#                     xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
#                     atom_idx = {"N": 0, "CA": 1, "C": 2}[atom_name]
#                     arr[res_seq-1, atom_idx] = torch.tensor(xyz)
#     return arr

# def folder_to_tensor(pdb_folder, max_res):
#     pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
#     tensors = [parse_pdb_backbone(os.path.join(pdb_folder, f), max_res) for f in pdb_files]
#     return torch.stack(tensors), pdb_files

# def parse_arguments():
#     parser = ap.ArgumentParser(description="Give specifications for the input data")
#     parser.add_argument(
#         "--pdb_folder",
#         type=str,
#         required=True,
#         help="path to the folder containing the pdb files."
#     )
#     parser.add_argument(
#         "--max_res",
#         type=int,
#         required=True,
#         help="number of residues of the protein, choose the maximum number of the residues that the protein has."
#     )
#     parser.add_argument(
#         "--output_file",
#         type=str,
#         default="xyz_training.pt",
#         help="Path to the output file storing xyz coordinates"
#     )
#     parser.add_argument(
#         "--pdb_name_file",
#         type=str,
#         default="pdb_files.pkl",
#         help="Path to the output file storing pdb names"
#     )
#     return parser.parse_args()
# # Usage:
# if __name__ == "__main__":
#     args = parse_arguments()
#     pdb_folder = args.pdb_folder
#     max_res = args.max_res
#     output_file = args.output_file
#     pdb_name_file = args.pdb_name_file
#     tensor_batch, pdb_names = folder_to_tensor(pdb_folder, max_res)
#     import pickle
#     torch.save(tensor_batch, output_file)
#     with open(pdb_name_file, "wb") as f:
#         pickle.dump(pdb_names, f)


# gmx trjconv -s /Users/prathamdhanoa/Downloads/Applications/RA/India/CSB_ML_in_MD/Scripts_KRAS_Struct_Analysis/MD_Data/2erl_A_DataSource/2erl_A_prod_R2.tpr -f /Users/prathamdhanoa/Downloads/Applications/RA/India/CSB_ML_in_MD/Scripts_KRAS_Struct_Analysis/MD_Data/2erl_A_DataSource/2erl_A_prod_R2_fit.xtc.xtc -o /Users/prathamdhanoa/Downloads/Applications/RA/India/CSB_ML_in_MD/Scripts_KRAS_Struct_Analysis/MD_Data/VAE_RF_Conf_Gen/all_pdb_structures.pdb

# When prompted, select the group you want (e.g., 0 for System, 1 for Protein)

import os
import torch
import argparse as ap
import pickle

def parse_pdb_backbone(pdb_path, max_res):
    """Parse single PDB file"""
    arr = torch.zeros(max_res, 3, 3)
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_seq = int(line[22:26].strip())
                if 1 <= res_seq <= max_res and atom_name in ["N", "CA", "C"]:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    atom_idx = {"N": 0, "CA": 1, "C": 2}[atom_name]
                    arr[res_seq-1, atom_idx] = torch.tensor(xyz)
    
    return arr

def parse_multi_model_pdb(pdb_path, max_res):
    """Parse single PDB file with multiple MODEL/ENDMDL blocks"""
    tensors = []
    model_names = []
    current_model = []
    model_count = 0
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                model_count = int(line.split()[1])
                current_model = []
            elif line.startswith("ENDMDL"):
                # Process accumulated model
                arr = torch.zeros(max_res, 3, 3)
                for atom_line in current_model:
                    atom_name = atom_line[12:16].strip()
                    res_seq = int(atom_line[22:26].strip())
                    if 1 <= res_seq <= max_res and atom_name in ["N", "CA", "C"]:
                        xyz = [float(atom_line[30:38]), float(atom_line[38:46]), float(atom_line[46:54])]
                        atom_idx = {"N": 0, "CA": 1, "C": 2}[atom_name]
                        arr[res_seq-1, atom_idx] = torch.tensor(xyz)
                tensors.append(arr)
                model_names.append(f"model_{model_count}")
            elif line.startswith("ATOM"):
                current_model.append(line)
    
    return torch.stack(tensors), model_names

def folder_to_tensor(pdb_folder, max_res):
    """Parse multiple PDB files from folder"""
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
    tensors = [parse_pdb_backbone(os.path.join(pdb_folder, f), max_res) for f in pdb_files]
    return torch.stack(tensors), pdb_files

def parse_arguments():
    parser = ap.ArgumentParser(description="Convert PDB to tensor format")
    
    parser.add_argument(
        "--pdb_folder",
        type=str,
        help="Path to folder containing multiple PDB files (use this OR --pdb_file)"
    )
    
    parser.add_argument(
        "--pdb_file",
        type=str,
        help="Path to single multi-model PDB file (use this OR --pdb_folder)"
    )
    
    parser.add_argument(
        "--max_res",
        type=int,
        required=True,
        help="Maximum number of residues"
    )
    
    parser.add_argument(
        "--output_train",
        type=str,
        default="xyz_training.pt",
        help="Output tensor training file"
    )

    parser.add_argument(
        "--output_test",
        type=str,
        default="xyz_test.pt",
        help="Output tensor test file"
    )
    
    parser.add_argument(
        "--pdb_name_file",
        type=str,
        default="pdb_files.pkl",
        help="Output file for PDB names"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.pdb_file:
        # Single multi-model PDB file
        tensor_batch, pdb_names = parse_multi_model_pdb(args.pdb_file, args.max_res)
    elif args.pdb_folder:
        # Multiple PDB files in folder
        tensor_batch, pdb_names = folder_to_tensor(args.pdb_folder, args.max_res)
    else:
        raise ValueError("Must provide either --pdb_file or --pdb_folder")
    
    torch.save(tensor_batch, args.output_train)
    with open(args.pdb_name_file, "wb") as f:
        pickle.dump(pdb_names, f)
    
    print(f"Saved {tensor_batch.shape[0]} structures to {args.output_train}")

    # l = output_file
    # for _ in range(No. of files in train data set//4):
    #   test_file =[]
    #   element = l[torch.rand_perm(1, no. of files in train data set)]
    #   with open file as 'wb':
    #       write element into test file
