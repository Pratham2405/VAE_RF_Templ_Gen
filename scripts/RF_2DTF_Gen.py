# import torch
# import argparse as ap

# PARAMS = {
#     "DMIN"    : 2.0,
#     "DMAX"    : 20.0,
#     "DBINS"   : 36,
#     "ABINS"   : 36,
# }

# # ============================================================
# def get_pair_dist(a, b):
#     """calculate pair distances between two sets of points
    
#     Parameters
#     ----------
#     a,b : pytorch tensors of shape [batch,nres,3]
#           store Cartesian coordinates of two sets of atoms
#     Returns
#     -------
#     dist : pytorch tensor of shape [batch,nres,nres]
#            stores paitwise distances between atoms in a and b
#     """

#     dist = torch.cdist(a, b, p=2)
#     return dist

# # ============================================================
# def get_ang(a, b, c):
#     """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
#     from Cartesian coordinates of three sets of atoms a,b,c 

#     Parameters
#     ----------
#     a,b,c : pytorch tensors of shape [batch,nres,3]
#             store Cartesian coordinates of three sets of atoms
#     Returns
#     -------
#     ang : pytorch tensor of shape [batch,nres]
#           stores resulting planar angles
#     """
#     v = a - b
#     w = c - b
#     eps = 1e-8
#     v /= torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
#     w /= torch.norm(w, dim=-1, keepdim=True).clamp_min(eps)
#     vw = torch.sum(v*w, dim=-1)

#     return torch.acos(vw.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

# # ============================================================
# def get_dih(a, b, c, d):
#     """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
#     given Cartesian coordinates of four sets of atoms a,b,c,d

#     Parameters
#     ----------
#     a,b,c,d : pytorch tensors of shape [batch,nres,3]
#               store Cartesian coordinates of four sets of atoms
#     Returns
#     -------
#     dih : pytorch tensor of shape [batch,nres]
#           stores resulting dihedrals
#     """
#     b0 = a - b
#     b1 = c - b
#     b2 = d - c

#     b1 /= torch.norm(b1, dim=-1, keepdim=True)

#     v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
#     w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

#     x = torch.sum(v*w, dim=-1)
#     y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

#     return torch.atan2(y, x)


# # ============================================================
# def xyz_to_c6d(xyz, params=PARAMS):
#     """convert cartesian coordinates into 2d distance 
#     and orientation maps
    
#     Parameters
#     ----------
#     xyz : pytorch tensor of shape [batch,nres,3,3]
#           stores Cartesian coordinates of backbone N,Ca,C atoms
#     Returns
#     -------
#     c6d : pytorch tensor of shape [batch,nres,nres,4]
#           stores stacked dist,omega,theta,phi 2D maps 
#     mask: pytorch boolean tensor of shape [batch,nres,nres]
#           True where orientations were computed (distance < DMAX)
#     """
    
#     batch = xyz.shape[0]
#     nres = xyz.shape[1]

#     # three anchor atoms
#     N  = xyz[:,:,0]
#     Ca = xyz[:,:,1]
#     C  = xyz[:,:,2]

#     # recreate Cb given N,Ca,C
#     b = Ca - N
#     c = C - Ca
#     a = torch.cross(b, c, dim=-1)
#     Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

#     # 6d coordinates order: (dist,omega,theta,phi)
#     c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

#     dist = get_pair_dist(Cb,Cb)
#     dist[torch.isnan(dist)] = 999.9
#     c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
#     b,i,j = torch.where(c6d[...,0]<params['DMAX'])

#     c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
#     c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
#     c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

#     # fix long-range distances
#     c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
#     mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
#     mask[b,i,j] = 1.0

#     # return boolean mask for clearer downstream use
#     return c6d, mask.to(torch.bool)

# def normalize_c6d(c6d_tensor):
#     normalized = torch.zeros_like(c6d_tensor)
    
#     # Per-channel normalization
#     for channel in range(4):
#         data = c6d_tensor[..., channel]
        
#         # For distance (channel 0), handle 999.9 sentinel values
#         if channel == 0:
#             mask = data < 999.0
#             valid_data = data[mask]
#             mean = valid_data.mean()
#             std = valid_data.std()
#             normalized[..., channel] = (data - mean) / std
#             normalized[..., channel][~mask] = 10.0  
#         else:
#             # Angles: simple standardization
#             mean = data.mean()
#             std = data.std()
#             normalized[..., channel] = (data - mean) / std
    
#     return normalized

# def c6d_parser():
#     parser = ap.ArgumentParser(description="Convert PDB xyz to c6d with mask")
#     parser.add_argument(
#         "--input_xyz",
#         type=str,
#         required=True,
#         help="Path to input .pt file containing xyz tensor"
#     )
#     parser.add_argument(
#         "--output_c6d",
#         type=str,
#         required=True,
#         help="Path to output .pt file to save c6d and mask tensors"
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = c6d_parser()
#     input_xyz = args.input_xyz
#     output_c6d = args.output_c6d
#     xyz = torch.load(input_xyz)
#     c6d, mask = xyz_to_c6d(xyz)
#     c6d = normalize_c6d(c6d)
#     torch.save({"c6d": c6d, "mask": mask}, output_c6d)

# Usage Examples:
# python RF_2DTF_Gen.py --input_xyz xyz_training.pt --output_c6d train_c6d_with_mask.pt

import torch
import argparse as ap
import os

PARAMS = {
    "DMIN": 2.0,
    "DMAX": 20.0,
    "DBINS": 36,
    "ABINS": 36,
}

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores pairwise distances between atoms in a and b
    """
    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c
    
    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    eps = 1e-8
    v /= torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    w /= torch.norm(w, dim=-1, keepdim=True).clamp_min(eps)
    vw = torch.sum(v*w, dim=-1)
    return torch.acos(vw.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d
    
    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 /= torch.norm(b1, dim=-1, keepdim=True)
    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1
    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    return torch.atan2(y, x)

# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps
    mask: pytorch boolean tensor of shape [batch,nres,nres]
          True where orientations were computed (distance < DMAX)
    """
    batch = xyz.shape[0]
    nres = xyz.shape[1]
    
    # three anchor atoms
    N = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C = xyz[:,:,2]
    
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    
    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4], dtype=xyz.dtype, device=xyz.device)
    
    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres, device=xyz.device)[None,...]
    
    b,i,j = torch.where(c6d[...,0] < params['DMAX'])
    c6d[b,i,j,1] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,2] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,3] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])
    
    c6d[c6d[...,0] >= params['DMAX']] = 999.9
    
    mask = torch.zeros((batch, nres, nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    
    return c6d, mask.to(torch.bool)

def c6d_parser():
    parser = ap.ArgumentParser(description="Convert PDB xyz to c6d with mask")

    parser.add_argument(
        "--input_xyz",
        type=str,
        required=True,
        help="Path to input .pt file containing xyz tensor"
    )

    parser.add_argument(
        "--output_c6d",
        type=str,
        required=True,
        help="Path to output .pt file to save c6d and mask tensors"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = c6d_parser()

    print(f"Loading xyz data from {args.input_xyz}...")
    xyz = torch.load(args.input_xyz)
    print(f"  Loaded {xyz.shape[0]} structures with {xyz.shape[1]} residues")

    print("Converting xyz to c6d...")
    c6d, mask = xyz_to_c6d(xyz)
    print(f"  C6D shape: {c6d.shape}")

    # Save c6d and mask without normalization
    torch.save({"c6d": c6d, "mask": mask}, args.output_c6d)
    print(f"Saved c6d and mask to {args.output_c6d}")
    print("Done!")

# Usage Example:
# python RF_2DTF_Gen.py --input_xyz xyz_data.pt --output_c6d output_c6d.pt
