import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, NamedTuple
from modules.quantize import Quantize
from modules.encoder import MLP
from einops import rearrange
from modules.loss import CategoricalReconstructionLoss, ReconstructionLoss

class SeqBatch(NamedTuple):
    user_ids : Tensor
    ids      : Tensor
    ids_fut  : Tensor
    x        : Tensor
    x_fut    : Tensor
    seq_mask : Tensor

class RqVaeOutput(NamedTuple):
    embeddings    : Tensor
    residuals     : Tensor
    sem_ids       : Tensor
    quantize_loss : Tensor

class RqVaeComputedLosses(NamedTuple):
    loss                : Tensor
    reconstruction_loss : Tensor
    rqvae_loss          : Tensor
    embs_norm           : Tensor
    p_unique_ids        : Tensor

class RqVae(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int,
                 hidden_dim: List[int],
                 codebook_size: int,
                 codebook_kmeans_init: bool = True,
                 codebook_normalize: bool = False,
                 n_layers: int = 3,
                 commitment_weight: float = 0.25,
                 n_cat_features: int = 18)->None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.commitment_weight = commitment_weight
        self.n_cat_features = n_cat_features

        self.layers = nn.ModuleList(
            modules=[
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    do_kmeans_init=codebook_kmeans_init,
                    commitment_weight=commitment_weight
                )
                for _ in range(n_layers)
            ]
        )

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim[-1::-1],
            out_dim=input_dim,
            normalize=False
        )

        self.reconstruction_loss = (
            CategoricalReconstructionLoss(n_cat_features)
            if n_cat_features!=0
            else ReconstructionLoss()
        )
    
    def load_pretrained(self, path: str)->None:
        device = next(self.parameters()).device
        state = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"-----Loaded RQVAE Iter {state['iter']}-----")

    def get_semantic_ids(self, x: Tensor, gumbel_t: float = 0.001):
        x = x.to(next(self.encoder.parameters()).dtype)
        res = self.encoder(x)

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized[2]
            emb, id = quantized[0], quantized[1]
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)
        
        print("Quantisation Successful")
        
        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss
        )
    
    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float)->RqVaeComputedLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decoder(embs.sum(axis=-1))
        x_hat = torch.cat(
            [F.normalize(x_hat[..., :-self.n_cat_features], p=2, dim=-1), x_hat[..., -self.n_cat_features:]],
            axis=-1
        )

        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        with torch.no_grad():
            embs_norm = embs.norm(axis=1)
            p_unique_ids = (
                -torch.triu(
                    (
                    rearrange(quantized.sem_ids, "b d -> b 1 d")
                    ==rearrange(quantized.sem_ids,"b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            rqvae_loss=rqvae_loss,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids
        )
