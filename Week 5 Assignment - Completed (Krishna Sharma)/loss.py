from torch import nn
from torch import Tensor

class ReconstructionLoss(nn.Module):
    def __init__(self)->None:
        super().__init__()
    
    def forward(self, x_hat: Tensor, x: Tensor)->Tensor:
        return ((x_hat - x)**2).mean(axis=-1)

class CategoricalReconstructionLoss(nn.Module):
    def __init__(self, n_cat_feats: int)->None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats
    
    def forward(self, x_hat: Tensor, x: Tensor)->Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:,:-self.n_cat_feats], x[:,:-self.n_cat_feats]
        )

        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:,-self.n_cat_feats:],
                x[:,-self.n_cat_feats:],
                reduction="none"
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr