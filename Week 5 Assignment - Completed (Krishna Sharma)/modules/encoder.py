from torch import Tensor
import torch.nn as nn
from typing import List
class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: List[int], 
                 out_dim: int, 
                 dropout: float=0.0, 
                 normalize: bool=False)->None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.normalize = normalize

        dims = [self.input_dim] + self.hidden_dim + [self.out_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i!= len(dims)- 2:
                self.mlp.append(nn.ReLU())
                if self.dropout != 0.0:
                    self.mlp.append(nn.Dropout(self.dropout))
    
    def forward(self, x: Tensor)->Tensor:

        z = self.mlp(x)
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=-1, eps=1e-12)
        
        return z

