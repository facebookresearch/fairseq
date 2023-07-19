'''
Credits: https://github.com/huggingface/transformers/issues/20403
'''

import torch.nn.functional as F 
import torch.nn as nn
import torch


class GLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.sigmoid(gate) * x

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

class ReGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.relu(gate) * x


class GLUFFN(nn.Module):
    def __init__(self, in_dim, hid_dim, act_fn='swiglu'):
        super().__init__()
        if act_fn == 'glu':
            self.act_fn = GLU()
        elif act_fn == 'reglu':
            self.act_fn = ReGLU()
        elif act_fn == 'geglu':
            self.act_fn = GEGLU()
        elif act_fn == 'swiglu':
            self.act_fn = SwiGLU()
        
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim//2, in_dim)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))