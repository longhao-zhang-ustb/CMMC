import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import MambaBlock, ModelArgs

class BiMambaEncoder(nn.Module):
    def __init__(self, mamba_args):
        super(BiMambaEncoder, self).__init__()
        self.mamba = MambaBlock(mamba_args)
        # Norm and feed-forward network layer
        self.f_norm = nn.LayerNorm(mamba_args.d_model)
        self.b_norm = nn.LayerNorm(mamba_args.d_model)
        self.s_norm = nn.LayerNorm(mamba_args.d_model)
        self.e_norm = nn.LayerNorm(mamba_args.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(mamba_args.d_model, mamba_args.d_model * 4),
            nn.GELU(),
            nn.Linear(mamba_args.d_model * 4, mamba_args.d_model)
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
