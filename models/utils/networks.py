# Imports
from typing import Optional
import torch
from torch import nn
from torch.nn.utils import weight_norm

# ~~~~~~~~~~~~~~~~~~~~~
# NETWORK ARCHITECTURE
# ~~~~~~~~~~~~~~~~~~~~~

def block(in_channels:int, out_channels:int, non_linearity:Optional[str]='relu', 
          norm_layer:Optional[str]='weigh_norm', dropout:float=0, bias:bool=True):
    
    if norm_layer == "weight_norm":
        layer = [weight_norm(nn.Linear(in_channels, out_channels, bias=bias))]
    elif norm_layer == "batch_norm":
        layer = [nn.Linear(in_channels, out_channels, bias=bias)]
        layer.append(nn.BatchNorm1d(out_channels))
    else:
        layer = [nn.Linear(in_channels, out_channels, bias=bias)]
        
    if non_linearity == 'relu':
        layer.append(nn.ReLU())
    elif non_linearity == 'leaky_relu':
        layer.append(nn.LeakyReLU())
    
    if dropout:
        layer.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layer)
# ======================================


class FasterDecoder(nn.Module):
    def __init__(self, config_model:dict, out_channels:int=1):
        super().__init__()
        
        self.shape_latent_size = config_model["shape_latent_size"]
        self.in_dim = config_model["in_dim"]
        self.hidden_layers = config_model["hidden_layers"]
        self.hidden_kernel_size = config_model["hidden_kernel_size"]  # kernel size (# features) in each hidden layer
        self.out_channels = out_channels
        self.norm_layer = config_model["norm_layer"]
        self.non_linearity = config_model["non_linearity"]
        self.dropout = config_model["dropout"]
         
        # Input layer
        layers = [block(in_channels=(self.shape_latent_size + self.in_dim), out_channels=self.hidden_kernel_size,
                        non_linearity=self.non_linearity, norm_layer=self.norm_layer, dropout=self.dropout)]
        # Hidden layers
        for _ in range(1, self.hidden_layers):
            layers.append(block(in_channels=self.hidden_kernel_size, out_channels=self.hidden_kernel_size,
                                non_linearity=self.non_linearity, norm_layer=self.norm_layer, dropout=self.dropout))
        # Output layer <------
        layers.append(block(in_channels=self.hidden_kernel_size, out_channels=self.out_channels,
                            non_linearity=None, norm_layer=self.norm_layer, dropout=self.dropout))
        
        self.layers = nn.Sequential(*layers)
    # ---------------------
        
    def forward(self, latent_z, coords):
        """
        Forward pass of the FasterDecoder network.
        
        Args:
            -- latent_z (torch.Tensor): Input batch of Shape Latents - (N, latent_size), where N=batch_size
            -- coords (torch.Tensor): Input batch of coordinates (N, #points, 3)

        Returns:
            -- x (torch.Tensor): Output batch of SDFs: (N, #points, 1) or (N, #points, 3)
        """
        
        # Note: .expand replicates values along the unqueezed dimension to match the target shape
        latent_z = latent_z[:,None,:].expand(-1, coords.shape[1], -1)               # (N, latent_size) --> (N, #points, latent_size)
        x = torch.cat((latent_z, coords), dim=2)                                    # Input batch: (N, #points, 3xyz + latent_size)
        y = self.layers(x)                                                          # Output batch: (N, #points, 1 or 2 or 3 sdfs)
        
        return y  

        