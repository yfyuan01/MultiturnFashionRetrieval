import torch
import torch.nn as nn

class ConCatModule(nn.Module):
    def __init__(self):
        super(ConCatModule,self).__init__()

    def forward(self, x):
        x = torch.cat(x,dim=1)
        return x