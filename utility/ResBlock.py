import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """ A Pytorch module to allow network layers to be easily packaged together in a 
        similar fashion to a nn.Sequential module but with a residual connection
        from the input allowing for the simple implementation of residual networks as 
        deserved in the paper "Deep Residual Learning for Image Recognition" by He et al."""
    def __init__(self,*args,res_transform=None):
        super().__init__()
        self.seq = nn.Sequential(*args)
        self.res_transform = res_transform

    def forward(self,x):
        if self.res_transform:
            x0 = self.res_transform(x)
        else:
            x0 = x
        x = self.seq(x)
        return x + x0
