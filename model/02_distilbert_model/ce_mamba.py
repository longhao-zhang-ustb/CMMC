import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class  CEMambaEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(CEMambaEncoder, self).__init__()
        self.mamba = Mamba(d_model=feature_dim)
        # Norm and feed-forward network layer
        self.f_norm = nn.LayerNorm(feature_dim)
        self.b_norm = nn.LayerNorm(feature_dim)
        self.s_norm = nn.LayerNorm(feature_dim)
        self.e_norm = nn.LayerNorm(feature_dim)
        # 这里原始的是4，现在调整为2看看效果
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        # Forward Mamba
        x_norm = self.s_norm(x)
        mamba_out_forward = self.mamba(x_norm)
        mamba_out_forward = mamba_out_forward + x_norm
        mamba_out_forward = self.f_norm(mamba_out_forward)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back
        mamba_out_backward = mamba_out_backward + x_norm
        mamba_out_backward = self.b_norm(mamba_out_backward)
        
        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        temp_mamba_out = mamba_out
        mamba_out = self.feed_forward(mamba_out)
        output = self.e_norm(mamba_out + temp_mamba_out)  
        return output
