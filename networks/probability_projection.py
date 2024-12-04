import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def compute_distance_map(lat_grid, lon_grid, lat0, lon0, device='cpu'):
    # Convert degrees to radians
    lat_grid_rad = torch.deg2rad(lat_grid).to(device)
    lon_grid_rad = torch.deg2rad(lon_grid).to(device)
    lat0_rad = torch.deg2rad(torch.tensor(lat0, device=device))
    lon0_rad = torch.deg2rad(torch.tensor(lon0, device=device))
    
    delta_lat = lat_grid_rad - lat0_rad
    delta_lon = lon_grid_rad - lon0_rad

    # Haversine formula
    a = torch.sin(delta_lat / 2)**2 + torch.cos(lat0_rad) * torch.cos(lat_grid_rad) * torch.sin(delta_lon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    R = 6371.0  # Earth's radius in kilometers
    D = R * c  # Distance in kilometers

    return D  # Shape: (720, 1440)

def compute_weights(distance_map, sigma=500):
    # distance_map: Shape (720, 1440)
    W = torch.exp(- (distance_map / sigma)**2)
    W = W / W.sum()
    return W  # Shape: (720, 1440)

def weighted_global_pooling(features, weights):
    # features: (batch_size, num_features, 720, 1440)
    # weights: (720, 1440)
    batch_size, num_features, H, W = features.shape
    # Reshape weights to (1, 1, H, W)
    weights = weights.unsqueeze(0).unsqueeze(0)
    # Multiply features by weights
    weighted_features = features * weights  # Broadcasting over batch and channels
    # Sum over spatial dimensions
    pooled_features = weighted_features.view(batch_size, num_features, -1).sum(dim=2)  # Shape: (batch_size, num_features)
    return pooled_features  # Shape: (batch_size, num_features)

class FeatureToOutput(nn.Module):
    def __init__(self, num_features, output_size):
        super(FeatureToOutput, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, output_size)
        )

    def forward(self, x):
        return self.fc(x)  # Shape: (batch_size, output_size)

class ProbabilityProjection(nn.Module):
    def __init__(self, num_features=20, sigma=500, output_size=1000, device='cpu'):
        super(ProbabilityProjection, self).__init__()
        self.num_features = num_features
        self.sigma = sigma
        self.output_size = output_size
        self.device = device

        # Define the feature mapping network
        self.feature_to_output = FeatureToOutput(num_features, output_size)

        # Precompute the latitude and longitude grids
        latitudes = torch.linspace(-90, 90, 720).unsqueeze(1).repeat(1, 1440)
        longitudes = torch.linspace(-180, 180, 1440).unsqueeze(0).repeat(720, 1)
        self.register_buffer('lat_grid', latitudes)  # Shape: (720, 1440)
        self.register_buffer('lon_grid', longitudes)  # Shape: (720, 1440)
    
    def forward(self, features, lat0, lon0):
        batch_size = features.shape[0]
        # Compute the distance map
        distance_map = compute_distance_map(self.lat_grid, self.lon_grid, lat0, lon0, device=features.device)  # Shape: (720, 1440)

        # Compute weights
        weights = compute_weights(distance_map, sigma=self.sigma)  # Shape: (720, 1440)

        # Compute weighted global pooling
        pooled_features = weighted_global_pooling(features, weights)  # Shape: (batch_size, num_features)

        # Map pooled features to output
        output = self.feature_to_output(pooled_features)  # Shape: (batch_size, output_size)

        # Optionally, apply softmax over the output dimension
        output = F.softmax(output, dim=1)

        return output  # Shape: (batch_size, output_size)
"""
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a sample input tensor
    batch_size = 1
    features = torch.randn(batch_size, 20, 720, 1440).to(device)

    # Define target coordinate (latitude, longitude)
    lat0 = 137.0   # Equator
    lon0 = 347.0   # Prime Meridian

    # Define the desired output size
    output_size = 1000

    # Initialize the probability projection module
    projection = ProbabilityProjection(num_features=20, sigma=500, output_size=output_size, device=device).to(device)

    # Forward pass
    t = time.time()
    output = projection(features, lat0, lon0)  # Shape: (batch_size, output_size)
    t =time.time() - t
    print('inference time = ', t)
    print('n params = ', sum(p.numel() for p in projection.parameters() if p.requires_grad))
    # Check output
    print(f"Output shape: {output.shape}")
    print(f"Sum over output (should be 1.0 for each sample): {output.sum(dim=1)}")
"""