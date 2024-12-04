import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import logging
from utils.YParams import YParams
from networks.afnonet import AFNONet
from utils.darcy_loss import LpLoss
from collections import OrderedDict
import h5py
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    """Generate random training data"""
    def __init__(self, params, num_samples=1000):
        self.params = params
        self.num_samples = num_samples
        self.img_shape_x = 720
        self.img_shape_y = 1440
        self.n_in_channels = params.N_in_channels
        self.n_out_channels = params.N_out_channels
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input and target with correct shapes
        inp = torch.randn(self.n_in_channels, self.img_shape_x, self.img_shape_y)
        tar = torch.randn(self.n_out_channels, self.img_shape_x, self.img_shape_y)
        return inp, tar

class Trainer:
    def __init__(self, params):
        self.params = params
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = AFNONet(
            params=params,
            img_size=(params.img_shape_x, params.img_shape_y),
            patch_size=(params.patch_size, params.patch_size),
            in_chans=params.N_in_channels,
            out_chans=params.N_out_channels,
            embed_dim=params.width,
            depth=params.depth,
            num_blocks=params.num_blocks
        ).to(self.device)
        
        # Setup data loaders
        self.train_dataset = RandomDataset(params, num_samples=1000)
        self.valid_dataset = RandomDataset(params, num_samples=100)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        if params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.max_epochs)
        else:
            self.scheduler = None
            
        self.loss_obj = LpLoss()
        self.epoch = 0
        
        print(f"Model has {self.count_parameters():,} trainable parameters")
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (inp, tar) in enumerate(self.train_loader):
            inp, tar = inp.to(self.device), tar.to(self.device)
            
            self.optimizer.zero_grad()
            gen = self.model(inp)
            loss = self.loss_obj(gen, tar)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch {self.epoch}, Batch {i+1}, Loss: {loss.item():.4f}, '
                      f'Time: {time.time() - start_time:.2f}s')
                start_time = time.time()
                
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inp, tar in self.valid_loader:
                inp, tar = inp.to(self.device), tar.to(self.device)
                gen = self.model(inp)
                loss = self.loss_obj(gen, tar)
                total_loss += loss.item()
                
        return total_loss / len(self.valid_loader)
    
    def train(self):
        best_valid_loss = float('inf')
        
        for epoch in range(self.params.max_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            train_loss = self.train_epoch()
            valid_loss = self.validate_epoch()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Save checkpoint if validation loss improved
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoint('best_model.pt')
                
            print(f'Epoch {epoch+1}/{self.params.max_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Time: {time.time() - start_time:.2f}s')
            print('-' * 80)
            
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoints', filename))

def main():
    # Setup parameters
    params = YParams('config/AFNO.yaml', 'afno_backbone')
    
    # Modify for local training
    params.batch_size = 1  # Small batch size for CPU/limited memory
    params.max_epochs = 5
    params.lr = 1e-4
    params.img_shape_x = 720
    params.img_shape_y = 1440
    params.patch_size = 8
    params.depth = 12
    params.num_blocks = 12
    params.width = 768
    params.N_in_channels = 20
    params.N_out_channels = 20
    params.scheduler = 'CosineAnnealingLR'
    
    # Initialize trainer and start training
    trainer = Trainer(params)
    trainer.train()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {str(e)}")