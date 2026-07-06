import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from modules.kmeans import Kmeans

def sample_gumbel(shape: Tuple, device: torch.device, eps=1e-20) -> Tensor:
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: Tensor, temperature: float, device: torch.device) -> Tensor:
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample

class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss

class Quantize(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 n_embed: int,
                 do_kmeans_init: bool = True,
                 commitment_weight: float = 0.25)->None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.do_kmeans_init = do_kmeans_init
        self.embedding = nn.Embedding(n_embed, embed_dim)

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self.kmeans_initted = False
        self._init_weights()
    
    @property
    def weights(self)->Tensor:
        return self.embedding.weight

    def _init_weights(self)->None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad
    def _kmeans_init(self, x)->None:
        with torch.no_grad():
            k, _ = self.embedding.weight.shape
            kmeans_out = Kmeans(k=k).run(x)
            self.embedding.weight.data.copy_(kmeans_out.centroids)
        self.kmeans_initted = True
    
    def get_item_embeddings(self, item_ids)->Tensor:
        return self.embedding(item_ids)
    
    def forward(self, x, temperature):

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)
        
        codebook = self.embedding.weight

        dist = ((x**2).sum(axis=1, keepdim=True) 
                + (codebook.T**2).sum(axis=0, keepdim=True)
                  - 2* x @ codebook.T)
        
        _, ids = (dist.detach()).min(axis=1)

        if self.training:
            weights = gumbel_softmax_sample(-dist, temperature, device=x.device)
            emb = weights @ codebook
            emb_out = emb

            loss = self.quantize_loss(query = x, value = emb_out)
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query = x, value = emb_out)
        
        return (emb_out, ids, loss)