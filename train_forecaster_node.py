import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import logging
from utils.YParams import YParams
from networks.afnonet import AFNONet, ProbabilitiesGenerator
from utils.darcy_loss import LpLoss
from collections import OrderedDict
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import uniform_, constant_, xavier_uniform_  # Fixed imports for initialization
from einops import rearrange
from functools import partial
from timm.models.layers import trunc_normal_
import boto3
from botocore.config import Config
from botocore.client import UNSIGNED
from tqdm import tqdm

import argparse
import os
from typing import Union, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#import wandb
from typing import Optional, Dict, Any



def compute_metrics(p, q):
    """
    Compute the three comparison metrics (accuracy, decisiveness, robustness) between two probability distributions.
    """
    # Ensure inputs are tensors
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    else:
        p = p.clone().detach()
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.clone().detach()
    
    # Ensure numerical stability
    eps = 1e-10
    p = torch.clamp(p, min=eps)
    p = p / p.sum()
    q = torch.clamp(q, min=eps)
    q = q / q.sum()
    
    # Compute metrics
    # Accuracy (Cross-Entropy)
    accuracy = torch.sum(p * torch.log(q)) 
    
    # Decisiveness (Arithmetic Mean)
    decisiveness = torch.sum(p * q)   # could be : torch.prod(torch.pow(q, -p))
    
    # Robustness (-2/3 Mean)
    r = -2/3
    sum_p_q_r = torch.sum(p * torch.pow(q, r))
    robustness = torch.pow(sum_p_q_r, 1 / r)   # could be : torch.prod(torch.pow(q, r*p))
    
    return {
        "accuracy": accuracy.item(),
        "decisiveness": decisiveness.item(),
        "robustness": robustness.item()
    }

"""class WandBLogger:
    def __init__(
        self,
        project_name: str,
        entity: str,
        api_key: str,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None
    ):
        
        # Set your API key as an environment variable
        os.environ["WANDB_API_KEY"] = api_key
        
        # Configure WandB to not require login
        os.environ["WANDB_MODE"] = "dryrun"
        
        # Initialize WandB run
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=run_name,
            tags=tags,
            anonymous="allow"  # Allows logging without authentication
        )
        
    def log(self, metrics: Dict[str, Any]):
        
        wandb.log(metrics)
        
    def finish(self):
        
        wandb.finish()"""

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

class PeriodicPad2d(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        # Assuming input is of shape (batch, channels, height, width)
        return F.pad(x, (self.pad,)*4, mode='circular')


class RandomDataset(Dataset):
    """Generate random training data"""
    def __init__(self, params, num_samples=1000):
        self.params = params
        self.num_samples = num_samples
        self.img_shape_x = 720
        self.img_shape_y = 1440
        self.n_in_channels = params.N_in_channels
        self.n_out_channels = params.output_size  # Changed to match probability output size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input
        inp = torch.randn(self.n_in_channels, self.img_shape_x, self.img_shape_y)
        # Generate random probability distribution target
        tar = torch.rand(self.n_out_channels)
        tar = tar / tar.sum()  # Normalize to create valid probability distribution
        return inp, tar
    


class ProbabilityTrainer:
    def __init__(self, params, backbone_checkpoint_path, freeze:bool=False):
        self.params = params
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.freeze_backbone = freeze
        # Initialize backbone model first
        self.backbone = AFNONet(
            params=params,
            img_size=(params.img_shape_x, params.img_shape_y),
            patch_size=(params.patch_size, params.patch_size),
            in_chans=params.N_in_channels,
            out_chans=params.N_out_channels,
            embed_dim=params.width,
            depth=params.depth,
            num_blocks=params.num_blocks
        ).to(self.device)
        
        # Initialize probability generator before loading backbone weights
        self.model = ProbabilitiesGenerator(
            params=params,
            backbone=self.backbone,
            output_size=params.output_size
        ).to(self.device)
        
        # Initialize head parameters with uniform distribution
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                with torch.no_grad():
                    uniform_(m.weight, -0.1, 0.1)
                    if m.bias is not None:
                        constant_(m.bias, 0)
        
        # Only initialize the head parameters
        self.model.conv.apply(init_weights)
        
        # Try to load backbone weights
        if not os.path.exists(backbone_checkpoint_path):
            print(f"Backbone checkpoint not found at {backbone_checkpoint_path}, attempting to download...")
            s3_bucket = "weather-data-20241123"
            s3_key = "model_weights/FCN_weights_v0/backbone.ckpt"
            
            if download_from_s3(s3_bucket, s3_key, backbone_checkpoint_path):
                print("Successfully downloaded backbone weights from S3")
            else:
                print(f"Failed to download backbone weights from s3://{s3_bucket}/{s3_key}. Training will proceed with random initialization.")
        
        # Load backbone weights with proper key mapping
        if os.path.exists(backbone_checkpoint_path):
            checkpoint = torch.load(backbone_checkpoint_path, map_location=self.device)
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
                
            # Remove the 'module.' prefix from state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
                new_state_dict[name] = v
                
            try:
                self.model.backbone.load_state_dict(new_state_dict)
                print("Successfully loaded pre-trained backbone weights from", backbone_checkpoint_path)
            except Exception as e:
                print(f"Error loading state dict: {str(e)}")
                print("Training will proceed with random initialization")
        
        # Initialize learnable weights for the metrics
        if hasattr(params, 'initial_weights'):
            self.metric_weights = nn.Parameter(params.initial_weights)
        else:
            self.metric_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.softmax = nn.Softmax(dim=0)
        print("Initial metric weights:", self.softmax(self.metric_weights))
        
        # Freeze backbone parameters
        if self.freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        # Setup data loaders
        train_dataset = DistributionDataset(
            params,
            num_samples=params.num_samples,
            dist_type=params.dist_type,
            **params.dist_params
        )
        valid_dataset = DistributionDataset(
            params,
            num_samples=max(100, params.num_samples // 10),  # 10% of training samples
            dist_type=params.dist_type,
            **params.dist_params
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize optimizer with only trainable parameters
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        trainable_params.append(self.metric_weights)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=params.lr)
        
        if params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.max_epochs)
        else:
            self.scheduler = None
        
        self.epoch = 0
        
        # Count parameters
        trainable_params = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.model.parameters()) + 3  # +3 for metric weights
        print(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")

    def compute_weighted_loss(self, pred_dist, target_dist):
        """Compute weighted sum of distribution comparison metrics"""
        # Ensure inputs are properly normalized
        pred_dist = F.softmax(pred_dist, dim=-1)
        target_dist = F.softmax(target_dist, dim=-1)
        
        # Get normalized weights
        weights = self.softmax(self.metric_weights)
        alpha, beta, gamma = weights
        
        # Compute metrics
        metrics = compute_metrics(target_dist, pred_dist)
        
        # Compute weighted loss (negative because we want to maximize these metrics)
        loss = -(alpha * metrics['accuracy'] + 
                beta * metrics['decisiveness'] + 
                gamma * metrics['robustness'])
        
        return loss, metrics
        
    def validate_epoch(self):
        """Run validation epoch"""
        self.model.eval()
        total_loss = 0
        metric_totals = {'accuracy': 0, 'decisiveness': 0, 'robustness': 0}
        last_batch_pred = None
        last_batch_target = None
        
        with torch.no_grad():
            for i, (inp, target_dist) in enumerate(self.valid_loader):
                inp, target_dist = inp.to(self.device), target_dist.to(self.device)
                pred_dist = self.model(inp)
                loss, metrics = self.compute_weighted_loss(pred_dist, target_dist)
                
                total_loss += loss.item()
                for k, v in metrics.items():
                    metric_totals[k] += v
                
                # Store last batch for visualization
                last_batch_pred = pred_dist[-1]
                last_batch_target = target_dist[-1]
        
        # Plot validation distributions
        if last_batch_pred is not None and last_batch_target is not None:
            plot_distributions(
                last_batch_target,
                F.softmax(last_batch_pred, dim=-1),
                self.epoch,
                len(self.valid_loader),
                output_dir='plots/validation'
            )
        
        # Average metrics over epoch
        n_batches = len(self.valid_loader)
        return (total_loss / n_batches,
                {k: v / n_batches for k, v in metric_totals.items()})


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        metric_totals = {'accuracy': 0, 'decisiveness': 0, 'robustness': 0}
        start_time = time.time()
        last_batch_pred = None
        last_batch_target = None
        
        for i, (inp, target_dist) in enumerate(self.train_loader):
            inp, target_dist = inp.to(self.device), target_dist.to(self.device)
            
            self.optimizer.zero_grad()
            pred_dist = self.model(inp)
            
            loss, metrics = self.compute_weighted_loss(pred_dist, target_dist)
            print(f'Loss: {loss.item():.4f}')
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in metrics.items():
                metric_totals[k] += v
            
            # Store last batch distributions for visualization
            last_batch_pred = pred_dist[-1].detach()  # Last item in batch
            last_batch_target = target_dist[-1].detach()
            
            if (i + 1) % 10 == 0:
                weights = self.softmax(self.metric_weights)
                print(f'Epoch {self.epoch}, Batch {i+1}:')
                print(f'Loss: {loss.item():.4f}')
                print(f'Metrics - Acc: {metrics["accuracy"]:.4f}, '
                    f'Decisiveness: {metrics["decisiveness"]:.4f}, '
                    f'Robustness: {metrics["robustness"]:.4f}')
                print(f'Weights - α: {weights[0]:.3f}, δ: {weights[1]:.3f}, ρ: {weights[2]:.3f}')
                print(f'Time: {time.time() - start_time:.2f}s\n')
                start_time = time.time()
        
        # Plot distributions from last batch
        if last_batch_pred is not None and last_batch_target is not None:
            plot_distributions(
                last_batch_target,
                F.softmax(last_batch_pred, dim=-1),
                self.epoch,
                len(self.train_loader)
            )
        
        # Average metrics over epoch
        n_batches = len(self.train_loader)
        return (total_loss / n_batches, 
                {k: v / n_batches for k, v in metric_totals.items()})




    def train(self):
        """Run full training process"""
        best_valid_loss = float('inf')
        
        print("\nStarting training...")
        for epoch in range(self.params.max_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self.train_epoch()
            
            # Validation phase
            valid_loss, valid_metrics = self.validate_epoch()
            
            # Step scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint if validation improved
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoint('best_probability_model.pt')
                print("New best model saved!")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch+1}/{self.params.max_epochs} completed in {epoch_time:.2f}s:')
            print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            print(f'Train Metrics: {train_metrics}')
            print(f'Valid Metrics: {valid_metrics}')
            weights = self.softmax(self.metric_weights)
            print(f'Current Metric Weights - α: {weights[0]:.3f}, β: {weights[1]:.3f}, γ: {weights[2]:.3f}')
            print('-' * 80)
        
        print("\nTraining completed!")


    

def generate_probability_distribution(dist_type: str, N: int, **params) -> torch.Tensor:
    """
    Generate a discrete probability distribution of size N based on the specified distribution type.
    
    Args:
        dist_type (str): Type of distribution ('normal', 'student_t', 'exponential')
        N (int): Size of the output distribution
        **params: Distribution-specific parameters
    
    Returns:
        torch.Tensor: Normalized probability distribution
    """
    # Generate appropriate x values for the distribution
    if dist_type == 'normal':
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        x = torch.linspace(mean - 4*std, mean + 4*std, steps=N)
        pdf = torch.exp(-0.5 * ((x - mean)/std)**2) / (std * np.sqrt(2 * np.pi))
    
    elif dist_type == 'student_t':
        df = params.get('df', 1.0)
        x = torch.linspace(-10, 10, steps=N)
        pdf = torch.tensor(stats.t.pdf(x.numpy(), df))
    
    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)
        x = torch.linspace(0, 5*scale, steps=N)
        pdf = torch.exp(-x/scale) / scale
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    # Normalize to create valid probability distribution
    pdf = torch.clamp(pdf, min=1e-10)  # Ensure numerical stability
    pdf = pdf / pdf.sum()
    
    return pdf, x

# Fix the DistributionDataset class
class DistributionDataset(Dataset):
    def __init__(self, params, num_samples=1000, dist_type='normal', **dist_params):
        self.params = params
        self.num_samples = num_samples
        self.img_shape_x = 720
        self.img_shape_y = 1440
        self.n_in_channels = params.N_in_channels
        self.n_out_channels = params.output_size
        self.dist_type = dist_type
        self.dist_params = dist_params
        
        # Set random parameters if not provided
        if dist_type == 'normal':
            if 'mean' not in dist_params:
                self.dist_params['mean'] = np.random.uniform(-2, 2)
            if 'std' not in dist_params:
                self.dist_params['std'] = np.random.uniform(0.5, 2)
        elif dist_type == 'student_t':
            if 'df' not in dist_params:
                self.dist_params['df'] = np.random.uniform(1, 5)
        elif dist_type == 'exponential':
            if 'scale' not in dist_params:
                self.dist_params['scale'] = np.random.uniform(0.5, 2)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input
        inp = torch.randn(self.n_in_channels, self.img_shape_x, self.img_shape_y)
        
        # Generate target distribution
        target_dist, _ = generate_probability_distribution(
            self.dist_type,
            self.n_out_channels,
            **self.dist_params
        )
        
        return inp, target_dist.float()

def plot_distributions(target_dist, pred_dist, epoch, batch, output_dir='plots'):
    """Plot target and predicted distributions"""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(target_dist))
    
    plt.plot(x, target_dist.cpu().numpy(), 'b-', label='Target Distribution', alpha=0.7)
    plt.plot(x, pred_dist.cpu().numpy(), 'r--', label='Predicted Distribution', alpha=0.7)
    
    plt.title(f'Distribution Comparison - Epoch {epoch}, Batch {batch}')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'dist_comparison_epoch_{epoch}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train probability distribution generator')
    parser.add_argument('--config', default='config/AFNO.yaml', help='Path to config file')
    parser.add_argument('--dataset_path', type=str, help='Path to training dataset (optional)')
    parser.add_argument('--dist_type', choices=['normal', 'student_t', 'exponential'], default='normal',
                      help='Type of distribution to generate if no dataset provided')
    parser.add_argument('--mean', type=float, help='Mean for normal distribution (optional)')
    parser.add_argument('--std', type=float, help='Standard deviation for normal distribution (optional)')
    parser.add_argument('--df', type=float, help='Degrees of freedom for student t distribution (optional)')
    parser.add_argument('--scale', type=float, help='Scale parameter for exponential distribution (optional)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--acc', type=float, default=1.0, help='Weight for accuracy metric')
    parser.add_argument('--dec', type=float, default=1.0, help='Weight for decisiveness metric')
    parser.add_argument('--rob', type=float, default=1.0, help='Weight for robustness metric')
    
    args = parser.parse_args()
    
    # Setup parameters
    params = YParams(args.config, 'afno_backbone')
    
    # Set training parameters
    params.batch_size = 1
    params.max_epochs = 5
    params.lr = 1e-4
    params.img_shape_x = 720
    params.img_shape_y = 1440
    params.patch_size = 8
    params.depth = 12
    params.num_blocks = 8  # To match checkpoint architecture
    params.width = 768
    params.N_in_channels = 20
    params.N_out_channels = 20
    params.output_size = 10  # Number of probability classes
    params.scheduler = 'CosineAnnealingLR'
    params.num_samples = args.num_samples
    
    # Set up distribution parameters
    params.dist_type = args.dist_type
    params.dist_params = {}
    if args.dist_type == 'normal':
        if args.mean is not None:
            params.dist_params['mean'] = args.mean
        if args.std is not None:
            params.dist_params['std'] = args.std
    elif args.dist_type == 'student_t':
        if args.df is not None:
            params.dist_params['df'] = args.df
    elif args.dist_type == 'exponential':
        if args.scale is not None:
            params.dist_params['scale'] = args.scale

    # Set up metric weights
    weights = torch.tensor([args.acc, args.dec, args.rob], dtype=torch.float32)
    if torch.any(weights < 0):
        raise ValueError("Metric weights must be non-negative")
    params.initial_weights = weights / weights.sum()  # Normalize to sum to 1
    
    print(f"Training configuration:")
    print(f"Distribution type: {params.dist_type}")
    print(f"Distribution parameters: {params.dist_params}")
    print(f"Normalized metric weights: α={params.initial_weights[0]:.3f}, "
          f"β={params.initial_weights[1]:.3f}, γ={params.initial_weights[2]:.3f}")
    
    # Create dataset
    if args.dataset_path:
        # Load dataset from file (implement this based on your data format)
        raise NotImplementedError("Loading from file not yet implemented")
    else:
        train_dataset = DistributionDataset(
            params,
            num_samples=params.num_samples,
            dist_type=params.dist_type,
            **params.dist_params
        )
    
    # Backbone checkpoint path
    backbone_checkpoint_path = 'checkpoints/best_model.pt'
    
    # Initialize trainer and start training
    trainer = ProbabilityTrainer(params, backbone_checkpoint_path)
    trainer.train()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception as e:
        print(f"Error during training: {str(e)}")



