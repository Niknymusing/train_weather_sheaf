import torch
from collections import OrderedDict
import logging
import os
import numpy as np
import time
import sys
from utils.YParams import YParams
from networks.afnonet import AFNONet


def load_model(model, params, checkpoint_file):
    """Load model weights with more robust state dict handling"""
    model.zero_grad()
    
    # Try loading with weights_only first
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"Warning: weights_only load failed, falling back to default load")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('backbone.'):
                key = key[9:]
            if key != 'ged':
                new_state_dict[key] = val
                
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except:
            model.load_state_dict(new_state_dict)
            
    except Exception as e:
        print(f"Failed to load with key modification, trying direct loading: {str(e)}")
        try:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        except Exception as e2:
            print(f"Direct loading failed as well: {str(e2)}")
            raise e2
            
    model.eval()
    return model

def setup(params, use_random_data=False):
    """Setup model and data"""
    device = 'cpu'  # Force CPU
    
    logging.info("Setting up model and data.")
    img_shape_x = 720
    img_shape_y = 1440
    n_in_channels = len(params.in_channels)
    n_out_channels = len(params.out_channels)
    
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.zeros((1, n_out_channels))
    params.stds = np.ones((1, n_out_channels))
    
    # Generate random test data
    valid_data_full = torch.randn(
        params.prediction_length, n_in_channels, img_shape_x, img_shape_y, 
        dtype=torch.float32
    ).numpy()
    
    # Load model
    model = AFNONet(params).to(device)
    model = load_model(model, params, params['best_checkpoint_path'])
    model.eval()
    
    return valid_data_full, model

if __name__ == "__main__":
    # Setup paths - adjust these to your local structure
    yaml_path = 'config/AFNO.yaml'  # Relative to current directory
    weights_path = os.path.expanduser('~/Downloads/backbone.ckpt')
    output_dir = 'output/'
    
    # Initialize params from YAML
    params = YParams(yaml_path, 'afno_backbone')  # Using backbone config instead of precip
    
    # Set required parameters
    params['world_size'] = 1
    params['global_batch_size'] = params.batch_size
    params['best_checkpoint_path'] = weights_path
    params['resuming'] = False
    params['local_rank'] = 0
    
    # Additional params that might be needed
    params.prediction_length = 4
    params.in_channels = list(range(20))  # [0-19] for backbone
    params.out_channels = list(range(20))
    
    print("Model configuration:")
    print(f"Input channels: {len(params.in_channels)}")
    print(f"Output channels: {len(params.out_channels)}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Get model and test data
        valid_data_full, model = setup(params, use_random_data=True)
        
        # Test inference
        x = torch.from_numpy(valid_data_full[0:1])  # Take first timestep and add batch dimension
        with torch.no_grad():
            t = time.time()
            output = model(x)
            print(output)
            inference_time = time.time() - t
            
            print("\nInference Results:")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Inference time: {inference_time:.3f} seconds")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
    except Exception as e:
        print(f"\nError during setup or inference: {str(e)}")
        print("\nPlease check:")
        print("1. Path to YAML config file (config/AFNO.yaml)")
        print("2. Path to backbone weights file (~/Downloads/backbone.ckpt)")
        print("3. All required dependencies are installed")
        sys.exit(1)