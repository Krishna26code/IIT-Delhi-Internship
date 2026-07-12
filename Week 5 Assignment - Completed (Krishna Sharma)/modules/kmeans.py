import numpy as np
import torch
from typing import NamedTuple, Optional


class KmeansOutput(NamedTuple):
    centroids: torch.Tensor
    assignment: Optional[torch.Tensor]


class Kmeans:
    def __init__(self,
                 k : int, 
                 max_iters : int=None,
                 stop_threshold : float=1e-10)->None:
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroid = None
        self.assignment = None
    
    def _init_centroid(self, x : torch.Tensor)->None:
        B, D = x.shape #D->Feature Dimension
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroid = x[init_idx, : ]
        self.assignment = None
    
    def _update_centroid(self, x : torch.Tensor)->torch.Tensor:
        squared_pw_dist = (
            x[:,None,:] - self.centroid[None,:,:]
        )**2 #Shape->[B, K, D]
        centroid_idx = (squared_pw_dist.sum(axis=2)).min(axis=1).indices
        assigned = (torch.arange(self.k, device=x.device).unsqueeze(1)==centroid_idx)

        for cluster in range(self.k):
            is_assigned_to_c = assigned[cluster]
            if not is_assigned_to_c.any():
                if x.size(0) > 0:
                    self.centroid[cluster, : ] = x[torch.randint(0,x.size(0),(1,))].squeeze(0)
                else:
                    raise ValueError("Can not choose random element from x, x is empty")
            else:
                self.centroid[cluster, : ] = x[is_assigned_to_c, : ].mean(axis=0)
        self.assignment = centroid_idx

    def run(self, x):
        self._init_centroid(x)

        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroid.clone()
            self._update_centroid(x)
            if torch.norm(self.centroid - old_c, dim=1).max() < self.stop_threshold:
                break
            i += 1
        
        return KmeansOutput(centroids=self.centroid, assignment=self.assignment)

