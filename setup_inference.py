import os
import boto3
import logging
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import torch
from torch import nn
from collections import OrderedDict
import numpy as np

# Get directory containing setup_inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths
yaml_path = os.path.join(BASE_DIR, 'config', 'AFNO.yaml')


from utils.YParams import YParams
from networks.afnonet import AFNONet

params = YParams(yaml_path, 'afno_backbone')
from networks.afnonet import ProbabilitiesGenerator  # Assuming it's saved in networks/




def load_model(model, params, checkpoint_file, projection_checkpoint_file=None):
    """Load model weights with robust state dict handling and optional projection layer weights."""
    model.zero_grad()

    # Load backbone weights
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {str(e)}")
        raise e

    # Load backbone weights
    backbone_state_dict = OrderedDict()
    for key, val in checkpoint['model_state'].items():
        if key.startswith('module.'):
            key = key[7:]
        if key.startswith('backbone.'):
            key = key[9:]
        backbone_state_dict[key] = val

    try:
        model.backbone.load_state_dict(backbone_state_dict, strict=False)
    except Exception as e:
        print(f"Failed to load backbone weights: {str(e)}")
        raise e

    # Load projection layer weights if provided
    if projection_checkpoint_file and os.path.exists(projection_checkpoint_file):
        try:
            projection_checkpoint = torch.load(projection_checkpoint_file, map_location='cpu')
            projection_state_dict = OrderedDict()
            for key, val in projection_checkpoint['model_state'].items():
                if key.startswith('module.'):
                    key = key[7:]
                if key.startswith('projection.'):
                    key = key[11:]
                projection_state_dict[key] = val
            model.load_state_dict(projection_state_dict, strict=False)
            print("Loaded projection layer weights from checkpoint.")
        except Exception as e:
            print(f"Failed to load projection layer weights: {str(e)}")
            # Initialize projection layer weights with Glorot initialization
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            print("Initialized projection layer weights with Glorot initialization.")
    else:
        # Initialize projection layer weights with Glorot initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print("Initialized projection layer weights with Glorot initialization.")

    model.eval()
    return model



def download_from_s3(bucket_name, s3_key, local_path):
    """Download file from S3 with progress bar."""
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Get file size
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        total_length = int(response.get('ContentLength', 0))
        
        # Download with progress bar
        with tqdm(total=total_length, unit='iB', unit_scale=True, desc=f'Downloading {s3_key}') as pbar:
            def callback(chunk):
                pbar.update(chunk)
            
            s3_client.download_file(
                bucket_name,
                s3_key,
                local_path,
                Callback=lambda bytes_transferred: callback(bytes_transferred)
            )
        
        logging.info(f"Successfully downloaded model checkpoint to {local_path}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to download model checkpoint: {str(e)}")
        return False

def setup_model():
    """Setup the ProbabilitiesProjection model with the AFNONet backbone."""
    device = 'cpu'  # Force CPU

    yaml_path = 'config/AFNO.yaml'  # Relative to current directory
    params = YParams(yaml_path, 'afno_backbone')  
    logging.info("Setting up model.")
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
    


    params['world_size'] = 1
    params['global_batch_size'] = params.batch_size
    params['resuming'] = False
    params['local_rank'] = 0
    # S3 configuration
    BUCKET_NAME = "weather-data-20241123"
    MODEL_KEY = "model_weights/FCN_weights_v0/backbone.ckpt"

    backbone_checkpoint = os.path.join("model_checkpoint", "backbone.ckpt")  # Local path where the model will be downloaded
    
    # Check if model checkpoint exists
    if not os.path.exists(backbone_checkpoint):
        logging.info(f"Model checkpoint not found at {backbone_checkpoint}. Downloading from S3...")
        success = download_from_s3(BUCKET_NAME, MODEL_KEY, backbone_checkpoint)
        if not success:
            raise RuntimeError("Failed to download model checkpoint")
    
    
    params['best_checkpoint_path'] = backbone_checkpoint
    params.output_size = 1000  # Desired output size for probability distribution


    # Initialize backbone
    backbone = AFNONet(params).to(device)

    # Initialize ProbabilitiesProjection model
    output_size = params.output_size  # Set this in your params
    model = ProbabilitiesGenerator(params, backbone, output_size).to(device)

    # Load model weights
    model = load_model(model, params, backbone_checkpoint, projection_checkpoint_file=None)
    model.eval()

    return model







"""
    

yaml_path = 'config/AFNO.yaml'  # Relative to current directory
backbone_checkpoint = os.path.expanduser('~/Downloads/backbone.ckpt')
projection_checkpoint = os.path.expanduser('~/Downloads/projection.ckpt')  # Optional
output_dir = 'output/'

    # Initialize params from YAML
params = YParams(yaml_path, 'afno_backbone')  # Using backbone config

    # Set required parameters
params['world_size'] = 1
params['global_batch_size'] = params.batch_size
params['resuming'] = False
params['local_rank'] = 0

    # Additional params
params.prediction_length = 4
params.in_channels = list(range(20))  # [0-19] for backbone
params.out_channels = list(range(20))
params.output_size = 1000  # Desired output size for probability distribution

print("Model configuration:")
print(f"Input channels: {len(params.in_channels)}")
print(f"Output channels: {len(params.out_channels)}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Setup the model
#model = setup_model(params, backbone_checkpoint, projection_checkpoint=projection_checkpoint)

    # Save model parameters if needed
model_params_path = os.path.join(output_dir, "model_params.pt")"""
#torch.save(model.state_dict(), model_params_path)


"""
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
"""
#afno_model = setup_afno_model(params, use_random_data=False)

#model_path = os.path.join(BASE_DIR, "saved_model.pt")

#if not os.path.exists(model_path):
#    model = setup_afno_model(params, use_random_data=False)

"""model_arch_path = os.path.join(BASE_DIR, "model_architecture.txt")
model_params_path = os.path.join(BASE_DIR, "model_params.pt")

if not os.path.exists(model_params_path):
    model = setup_afno_model(params, use_random_data=False)
    # Save architecture class name
    with open(model_arch_path, 'w') as f:
        f.write('AFNONet')
    # Save model parameters
    torch.save(model.state_dict(), model_params_path)

"""

    #torch.save(model, model_path)
    #print('AFNONet model savet to ', model_path)





def setup_afno_model(params, use_random_data=False):
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
    if use_random_data:
        valid_data_full = torch.randn(
            params.prediction_length, n_in_channels, img_shape_x, img_shape_y, 
            dtype=torch.float32
        ).numpy()
    
    # Load model
        model = AFNONet(params).to(device)
        model = load_model(model, params, params['best_checkpoint_path'])
        model.eval()
        
        return valid_data_full, model
    else:
        model = AFNONet(params).to(device)
        model = load_model(model, params, params['best_checkpoint_path'])
        model.eval()

        return model





"""import os
import sys
from utils.YParams import YParams
from networks.afnonet import AFNONet

# Get the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the YAML file relative to the script's directory
yaml_path = os.path.join(BASE_DIR, 'configs', 'AFNO.yaml')  # Fixed the folder name to 'configs'

# Verify the YAML file path
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"YAML file not found at: {yaml_path}")

# Initialize the parameters
params = YParams(yaml_path, 'afno_backbone')

"""

"""import os
import sys
import torch
import numpy as np
from collections import OrderedDict

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from FourCastNet.utils.YParams import YParams
from FourCastNet.networks.afnonet import AFNONet

# Get directory containing setup_inference.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, 'config', 'AFNO.yaml')

#from .utils.YParams import YParams
#from .networks.afnonet import AFNONet

params = YParams(yaml_path, 'afno_backbone')

"""