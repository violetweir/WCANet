import torch
from fast_wtconv.wtconv import WTConv2d

# Initialize layer
# in_channels, out_channels, kernel_size, stride, wt_levels
layer = WTConv2d(64, 64, kernel_size=5, wt_levels=2)

# Move to device (CUDA or MPS)
device = 'cuda' 
layer = layer.to(device)

# Forward pass
x = torch.randn(1, 64, 224, 224).to(device)
output = layer(x)