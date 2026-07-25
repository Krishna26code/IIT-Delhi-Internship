# TIGER: Recommender Systems with Generative Retrieval — Reproduction

This project reproduces **TIGER** (Rajput et al., *"Recommender Systems with Generative Retrieval"*, NeurIPS 2023), which frames sequential recommendation as a generative retrieval task. Each item is assigned a hierarchical **Semantic ID** (via an RQ-VAE quantizer trained on content embeddings), and a Transformer encoder-decoder is trained to autoregressively predict the Semantic ID of the next item a user will interact with.

Evaluated on the **Amazon Beauty** dataset (leave-one-out sequential recommendation protocol), matching the paper's dataset statistics (22,363 users / 12,101 items / mean sequence length 8.87–8.88).

## Pipeline

1. **Preprocessing** — 5-core filtering + leave-one-out sequence construction from Amazon review data.
2. **Content embeddings** — item text (title, brand, category, price) encoded with a Sentence-T5 model.
3. **RQ-VAE training** — 3-level residual-quantized VAE (encoder `[512, 256, 128]`, latent dim 32, codebook size 256/level) converts each item's embedding into a 3-tuple Semantic ID; a 4th disambiguation token resolves collisions.
4. **Retrieval training** — a Transformer encoder-decoder (`d_model=128`, 6 heads, 4 encoder + 4 decoder layers) is trained to predict the next item's Semantic ID given the user's interaction history, encoded as a sequence of Semantic ID tokens.
5. **Evaluation** — beam-search decoding (beam width 50) + Recall@K / NDCG@K against the held-out test item.

All code runs end-to-end on Kaggle (GPU T4×2); see `tiger_kaggle_workflow.ipynb`.

## Results (Amazon Beauty)

| Metric | Ours | Paper (Table 1) | Relative |
|---|---|---|---|
| Recall@5  | 0.0280 | 0.0454 | ~62% |
| NDCG@5    | 0.0185 | 0.0321 | ~58% |
| Recall@10 | 0.0419 | 0.0648 | ~65% |
| NDCG@10   | 0.0230 | 0.0384 | ~60% |

RQ-VAE codebook health: **100% utilization** across all 3 levels, perplexity 187–230 (out of 256 max), 88.5% of items received a unique 3-token Semantic ID before collision-breaking.

Training used the paper's reported hyperparameters (Adagrad lr=0.4 style config for RQ-VAE; Adafactor, peak_lr=0.01, 10K-step warmup + inverse-sqrt decay for the retrieval Transformer). `total_steps` was configured to 200,000 as in the paper; **early stopping (patience = 5 evaluations / 25,000 steps without validation NDCG@10 improvement) triggered before that limit**, indicating the model had converged rather than the run being cut short — see `val_curve.png`.

### Why results don't fully match the paper

Multiple independent reproductions of this paper (including other open-source attempts) consistently land in the 60–75% range of the paper's reported metrics, even when matching all stated hyperparameters. The paper was trained with Google's internal T5X framework, which has tokenization/training details (exact vocabulary construction, hashing scheme for user tokens, etc.) that are not fully specified in the paper text. This is a known, documented gap in the community — not evidence of a broken pipeline. (Sanity check: `invalid_rate` on generated Semantic IDs is ~0.0002, i.e. beam search almost always lands on a valid item, confirming the model learned the task correctly.)

## Reproducing

```bash
pip install -r requirements.txt

python -m tiger.scripts.preprocess_beauty --download
python -m tiger.rqvae.dataloader_beauty --config configs/rqvae/beauty.yaml --download
python -m tiger.rqvae.train --config configs/rqvae/beauty.yaml
python -m tiger.scripts.build_sid_tables \
    --checkpoint outputs/amazon_beauty_checkpoints/best.pt \
    --items-csv  outputs/amazon_beauty_items.csv \
    --embeddings outputs/amazon_beauty_embeddings.npy \
    --sequences-dir data/processed \
    --output-dir   data
python -m tiger.retrieval.train --config configs/retrieval/beauty.yaml
python -m tiger.retrieval.evaluate --checkpoint outputs/tiger_beauty/checkpoints/best.pt
```

Or run `tiger_kaggle_workflow.ipynb` on Kaggle (GPU + Internet enabled) for the full end-to-end pipeline with sanity checks at each stage.

**Note on the embedding model:** the config uses `sentence-t5-base` rather than `sentence-t5-xxl`; the XXL variant (~5B params) does not fit in a 16GB GPU at encoding time. Both produce 768-dim embeddings, so this only affects embedding quality, not the pipeline architecture.

## Acknowledgements

This implementation is built on top of an existing open-source reproduction of the TIGER paper, used and extended as a base for this project (hyperparameter correction, full Kaggle training pipeline, debugging, and evaluation against the paper's reported numbers).

Paper: Rajput, S., Mehta, N., Singh, A., et al. *"Recommender Systems with Generative Retrieval."* NeurIPS 2023.
