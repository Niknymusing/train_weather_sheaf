import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
import numpy as np
from utils.YParams import YParams
from networks.afnonet import AFNONet

# Load parameters from config
params = YParams('config/AFNO.yaml', 'afno_backbone')

# Add required parameters to params
params.N_in_channels = 20
params.N_out_channels = 20
params.patch_size = 8  # From YAML config
params.num_blocks = 8  # From YAML config
params.img_size = (720, 1440)  # Correct size for global weather data: 720 lat x 1440 lon

# Create random input data matching the expected size
batch_size = 1
random_input = np.random.randn(
    batch_size, 
    params.N_in_channels, 
    params.img_size[0],  # 720 latitude points
    params.img_size[1]   # 1440 longitude points
).astype(np.float32)

# Convert to tensor
input_tensor = torch.from_numpy(random_input)

# Initialize model with params
model = AFNONet(
    params=params,
    embed_dim=params.width,  # From YAML: width parameter
)

# Move to device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model = model.to(device)
input_tensor = input_tensor.to(device)

# Run inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
