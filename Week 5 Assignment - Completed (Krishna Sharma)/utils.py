import argparse
import torch

from modules.semids import TokenizedSeqBatch

def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner

@torch.no_grad
def compute_debug_metrics(batch: TokenizedSeqBatch, model_output=None, prefix: str = "")->dict:
    seq_lengths = batch.seq_mask.sum(axis=-1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix + f"seq_lengths_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item()
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item()
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics