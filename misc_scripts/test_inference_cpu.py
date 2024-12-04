import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
import numpy as np
import h5py
from inference.inference import setup, autoregressive_inference
from utils.YParams import YParams
from networks.afnonet import AFNONet

class ModelParams:
    """Wrapper class to ensure all parameters are set correctly"""
    def __init__(self):
        self.N_in_channels = 20
        self.N_out_channels = 20
        self.patch_size = 8
        self.num_blocks = 8
        self.width = 56  # embedding dimension
        self.modes = 32
        self.img_size = (720, 1440)
        self.embed_dim = 56  # Must match width
        self.depth = 8      # Must match num_blocks
        self.mlp_ratio = 4  # For correct MLP sizes
        self.sparsity_threshold = 0.01
        self.block_size = 7  # For 7x7 filters
        self.hidden_size_factor = 1
        self.hidden_size = self.width
        self.hard_thresholding_fraction = 1.0

# Create test directory structure
test_dir = "test_inference"
test_data_dir = os.path.join(test_dir, "test_data")
os.makedirs(test_data_dir, exist_ok=True)

# Create test data
test_shape = (10, 20, 720, 1440)
with h5py.File(os.path.join(test_data_dir, "test.h5"), 'w') as f:
    dset = f.create_dataset("fields", 
                           shape=test_shape,
                           dtype='float32',
                           chunks=True,
                           compression="gzip")
    for i in range(0, test_shape[0], 2):
        end_idx = min(i + 2, test_shape[0])
        chunk_shape = (end_idx - i,) + test_shape[1:]
        dset[i:end_idx] = np.random.randn(*chunk_shape).astype(np.float32)

# Create means and stds
np.save(os.path.join(test_data_dir, "global_means.npy"), 
        np.zeros((1, 20, 1, 1), dtype=np.float32))
np.save(os.path.join(test_data_dir, "global_stds.npy"), 
        np.ones((1, 20, 1, 1), dtype=np.float32))
np.save(os.path.join(test_data_dir, "time_means.npy"), 
        np.zeros((1, 20, 720, 1440), dtype=np.float32))

# Initialize model parameters
model_params = ModelParams()

# Initialize model with correct dimensions
model = AFNONet(
    params=model_params,
    img_size=model_params.img_size,
    patch_size=(model_params.patch_size, model_params.patch_size),
    in_chans=model_params.N_in_channels,
    out_chans=model_params.N_out_channels,
    embed_dim=model_params.width,  # Use width (56) for embedding
    depth=model_params.depth,      # 8 blocks
    mlp_ratio=model_params.mlp_ratio,  # 4 for correct MLP sizes
    num_blocks=model_params.num_blocks,  # Match filter sizes
    sparsity_threshold=model_params.sparsity_threshold,
    hard_thresholding_fraction=model_params.hard_thresholding_fraction
)

# Create and save checkpoint
checkpoint_dir = os.path.join(test_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = {
    'model_state': model.state_dict(),
    'optimizer_state': None,
    'epoch': 0
}
checkpoint_path = os.path.join(checkpoint_dir, 'best_ckpt.tar')
torch.save(checkpoint, checkpoint_path)

# Set up parameters for inference
params = YParams('config/AFNO.yaml', 'afno_backbone')
params.N_in_channels = model_params.N_in_channels
params.N_out_channels = model_params.N_out_channels
params.patch_size = model_params.patch_size
params.num_blocks = model_params.num_blocks
params.width = model_params.width
params.img_size = model_params.img_size
params.modes = model_params.modes
params.nettype = 'afno'

# Add paths and other parameters
params.inf_data_path = test_data_dir
params.experiment_dir = test_dir
params['best_checkpoint_path'] = checkpoint_path
params.dt = 1
params.n_history = 0
params.prediction_length = 4
params.prediction_type = 'iterative'
params.n_initial_conditions = 1
params.batch_size = 1
params.interp = 0
params.log_to_screen = True
params.masked_acc = False
params.use_daily_climatology = False
params.perturb = False
params.orography = False

params.global_means_path = os.path.join(test_data_dir, "global_means.npy")
params.global_stds_path = os.path.join(test_data_dir, "global_stds.npy")
params.time_means_path = os.path.join(test_data_dir, "time_means.npy")

print("Setting up model...")
valid_data_full, model = setup(params)
print("Running inference...")
ic = 0
sr, sp, vl, a, au, vc, ac, acu, accland, accsea = autoregressive_inference(params, ic, valid_data_full, model)

print("Inference completed successfully!")
print("Prediction shape:", sp.shape)
print("RMSE shape:", vl.shape)