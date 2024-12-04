import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Conv2DToLatentNet(nn.Module):
    def __init__(self, input_channels=20, output_size=1000):
        """
        Initialize the 2D convolutional network.
        Args:
            input_channels (int): Number of input channels (20 in this case).
            output_size (int): Size of the output probability distribution.
        """
        super(Conv2DToLatentNet, self).__init__()
        
        # Compute latent_dim based on output_size
        self.latent_dim = self.compute_latent_dim(output_size)
        print(f"Computed latent_dim: {self.latent_dim}")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        
        # Calculate the flattened size after convolutions
        self._compute_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, output_size)
        
    def compute_latent_dim(self, output_size):
        """
        Compute the latent dimension as the closest multiple of 128 to 4 * output_size.
        """
        target_dim = 4 * output_size
        latent_dim = 128 * round(target_dim / 128)
        return latent_dim
    
    def _compute_flattened_size(self):
        """
        Compute the flattened size after convolutions for initialization.
        """
        # Create a dummy input tensor with the appropriate size
        dummy_input = torch.zeros(1, 20, 720, 1440)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        self.flattened_size = x.numel()
        
    def forward(self, x):
        # x shape: (batch_size, 20, 720, 1440)
        x = F.relu(self.conv1(x))  # First convolution
        x = F.relu(self.conv2(x))  # Second convolution
        x = F.relu(self.conv3(x))  # Third convolution
        x = F.relu(self.conv4(x))  # Fourth convolution
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # Latent representation
        x = self.fc2(x)          # Final output layer
        x = F.softmax(x, dim=1)  # Apply softmax for probability distribution
        return x

# Example usage:
# Define the network with desired output size

# Forward pass through the network
#output = model(input_tensor)

# Output shape should be (batch_size, output_size)
#print(f"Output shape: {output.shape}")  # Should print: torch.Size([4, 1000])

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, -kept_modes:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, -kept_modes:, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, -kept_modes:, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, -kept_modes:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, -kept_modes:, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, -kept_modes:, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, -kept_modes:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, -kept_modes:, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, -kept_modes:, :kept_modes], self.w2[1]) +
            self.b2[0]
        )

        o2_imag[:, -kept_modes:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, -kept_modes:, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, -kept_modes:, :kept_modes], self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias

class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(
            hidden_size=dim,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(720, 1440), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        return x

class AFNONetProbabilityDecoder(nn.Module):
    def __init__(
            self,
            params,
            out_dim =1024,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=3,
            out_chans=1,  # Output channels set to 1
            embed_dim=512,  # Adjusted embed_dim
            depth=12,
            mlp_ratio=4.0,  # Adjusted mlp_ratio
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim
        )
        
        self.out_dim = out_dim
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                num_blocks=self.num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction
            ) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Modified head with adjusted hidden dimension to meet parameter requirements
        hidden_dim = 8192 #32768  # Adjusted hidden_dim to achieve desired parameter count

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, self.patch_size[0] * self.patch_size[1], bias=False)
        )

        self.out_proj = nn.Linear(720*1440, self.out_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # Shape: (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x  # Shape: (B, h, w, embed_dim)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.head(x)  # Shape: (B, h, w, patch_size[0]*patch_size[1])
        x = rearrange(
            x,
            "b h w (p1 p2) -> b 1 (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.h,
            w=self.w,
        )  # Shape: (B, 1, H, W)
        x = x.view(x.size(0), -1)  # Flatten to (B, H*W)
        x = self.out_proj(x)
        x = torch.softmax(x, dim=1)  # Apply softmax over spatial dimensions
        return x  # Shape: (B, H*W)






if __name__ == "__main__":
    # Define parameters
    class Params:
        patch_size = 16
        N_in_channels = 3
        N_out_channels = 1  # Output channels set to 1
        num_blocks = 16

    params = Params()

    # Initialize the AFNONetProbabilityDecoder model
    model_p = AFNONetProbabilityDecoder(
        params=params,
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=3,
        out_chans=1,
        embed_dim=128,        # Adjusted embed_dim
        depth=4,
        mlp_ratio=4.0,        # Adjusted mlp_ratio
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0
    )

    # Create a sample input tensor
    batch_size = 1  # Adjust based on your hardware capabilities
    input_tensor = torch.randn(batch_size, 3, 720, 1440)

    # Forward pass
    t = time.time()
    probability_distribution = model_p(input_tensor)
    t = time.time() - t
    print('inference time = ', t)
    print('n params = ', sum(p.numel() for p in model_p.parameters() if p.requires_grad))
    # Check output
    print(f"Probability distribution shape: {probability_distribution.shape}")  # Should be (batch_size, 720*1440)
    print(f"Sum over probabilities (should be 1.0 for each sample): {probability_distribution.sum(dim=1)}")
