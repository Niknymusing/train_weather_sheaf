import os
import sys
import time
import logging
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict
from utils.YParams import YParams
from networks.afnonet import AFNONet

class RandomDataset:
    """Generate random training data on the fly"""
    def __init__(self, params, num_samples=100):
        self.params = params
        self.num_samples = num_samples
        self.img_shape_x = 720
        self.img_shape_y = 1440
        self.n_in_channels = len(params.in_channels)
        self.n_out_channels = len(params.out_channels)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target
        inp = torch.randn(self.n_in_channels, self.img_shape_x, self.img_shape_y)
        target = torch.randn(self.n_out_channels, self.img_shape_x, self.img_shape_y)
        return inp, target

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, '
                  f'Time: {time.time() - start_time:.2f}s')
            start_time = time.time()
    
    return total_loss / len(train_loader)

def setup_training():
    """Initialize all training components"""
    # Setup paths
    yaml_path = 'config/AFNO.yaml'
    output_dir = 'output/train_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize params from YAML
    params = YParams(yaml_path, 'afno_backbone')
    
    # Set training parameters
    params.batch_size = 2  # Small batch size for CPU
    params.max_epochs = 5
    params.learning_rate = 1e-3
    params.in_channels = list(range(20))
    params.out_channels = list(range(20))
    
    # Model parameters
    params.N_in_channels = len(params.in_channels)
    params.N_out_channels = len(params.out_channels)
    params.embed_dim = 768
    params.depth = 12
    params.num_blocks = 12
    
    # Create random dataset and dataloader
    train_dataset = RandomDataset(params, num_samples=100)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=0  # No multiprocessing for CPU
    )
    
    # Initialize model
    device = 'cpu'
    model = AFNONet(
        params=params,
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=params.N_in_channels,
        out_chans=params.N_out_channels,
        embed_dim=params.embed_dim,
        depth=params.depth,
        num_blocks=params.num_blocks,
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=params.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=params.max_epochs)
    criterion = torch.nn.MSELoss()
    
    return model, train_loader, optimizer, scheduler, criterion, device, params, output_dir

def train():
    """Main training loop"""
    print("Setting up training...")
    model, train_loader, optimizer, scheduler, criterion, device, params, output_dir = setup_training()
    
    print("\nTraining configuration:")
    print(f"Device: {device}")
    print(f"Batch size: {params.batch_size}")
    print(f"Learning rate: {params.learning_rate}")
    print(f"Max epochs: {params.max_epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(params.max_epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{params.max_epochs}")
        print(f"Average loss: {train_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': train_loss,
            }
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")
    return model

if __name__ == "__main__":
    try:
        trained_model = train()
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)