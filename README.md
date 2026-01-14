# VAE_RF_Conf_Gen
Protein Conformation Generation via replication of Mansoor, Baker et al., 2024. Using Vanilla VAE and RosettaFold Integration
## Steps of the Workflow:
- Clone this repository:
```
git clone https://github.com/Pratham2405/VAE_RF_Conf_Gen.git
cd VAE_RF_Conf_Gen
cd scripts
```
- Create and activate a virtual environment. This creates a folder `vae_env` in the repo's directory which will contain your venv executables. In case you can't find it, use `find "$HOME" -name "activate" -type f`. This shows the path of the venv directory. To activate it, simply copy the path and give the command: `source "absolute/path/to/your/venv/bin/activate"`:
```
python -m venv vae_env
source vae_env/bin/activate
```
- Requirements.txt: Install the dependencies of this repo with:
```
pip install -r requirements.txt
```
- Dataset Download: Choose the dataset from the ATLAS database and split into individual `.pdb` files. Choose different folders for training and test `.pdb` folders(a ratio of 4:1 is considered standard).
- Run `python3 Prep_PDB42DTF.py`: this script extracts the 3D coordinates of the backbone atoms of the `.pdb` files in a folder and stacks them in a torch tensor with `torch.save`(repeat this for both training and test folders):
```
python3 Prep_PDB42DTF.py \
  --pdb_folder /path/to/pdb_folder \
  --max_res 40 \
  --output_file xyz_training.pt \
  --pdb_name_file pdb_files.pkl
```
- Generate c6d + mask from xyz tensor:
```
python -c "import torch; from RF_2DTF_Gen import xyz_to_c6d, PARAMS; \
xyz = torch.load('xyz_training.pt'); \
c6d, mask = xyz_to_c6d(xyz); \
torch.save({'c6d': c6d, 'mask': mask, 'params': PARAMS}, 'train_c6d_with_mask.pt')"
```
Repeat the above two steps for the test folder as well. Remember to rename the output file as `xyz_test.pt` and so on downstream. `c6d_sanity_check.py` prints the number of structures in the tensor for verifying that the right file has been prepared.

- Train the VAE. For a more efficient fine-tuning experience, you can choose the following arguments in the script:
```
python main.py \
  --protein_length 40 \
  --training_data train_c6d_with_mask.pt \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 1e-3 \
  --kl_div_weight 1e-3 \
  --feature_dim 64 \
  --latent_dim 256 \
  --num_workers 4 \
  --pin_memory True \
  --log_every 50 \
  --output_weights vae_weights.pth
```
- The output weigths will be saved to `vae_weights.pth`. You can choose otherwise. The values chosen for the above arguments are the default values except for `--protein_length` and `training_data` which are required fields. 
- A `vae_ckpt.pt` checkpoint file will be saved for storing the training parameter of this run. Tag the ckpt file for identification to avoid confusion in `main.py`. The new ckpt file created in a new training instance would replace the previous one so you would not have a record of the older parameters.
- The sampling/evaluation script, `sampling.py` offers you a choice between prior and posterior sampling. Both sampling options lead to a histogram for comparing the distributions of the 4 channels and reconstruction loss between the generated samples and the test dataset or the training dataset in the case of prior sampling.
