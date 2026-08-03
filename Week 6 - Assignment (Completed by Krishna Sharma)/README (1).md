# TIGER: Recommender Systems with Generative Retrieval — Reproduction

This project reproduces **TIGER** (Rajput et al., *"Recommender Systems with Generative Retrieval"*, NeurIPS 2023), which frames sequential recommendation as a generative retrieval task. Each item is assigned a hierarchical **Semantic ID** (via an RQ-VAE quantizer trained on content embeddings), and a Transformer encoder-decoder is trained to autoregressively predict the Semantic ID of the next item a user will interact with.

Evaluated on all three of the paper's benchmark subsets of the Amazon Product Reviews dataset — **Beauty**, **Sports and Outdoors**, and **Toys and Games** (leave-one-out sequential recommendation protocol) — matching the paper's dataset statistics exactly on every dataset:

| Dataset | Users (ours / paper) | Items (ours / paper) | Mean seq. len. (ours / paper) | Median seq. len. (ours / paper) |
|---|---|---|---|---|
| Beauty | 22,363 / 22,363 | 12,101 / 12,101 | 8.88 / 8.87 | 6 / 6 |
| Sports and Outdoors | 35,598 / 35,598 | 18,357 / 18,357 | 8.32 / 8.32 | 6 / 6 |
| Toys and Games | 19,412 / 19,412 | 11,924 / 11,924 | 8.63 / 8.63 | 6 / 6 |

## Pipeline

1. **Preprocessing** — 5-core filtering + leave-one-out sequence construction from Amazon review data. Run once per dataset (`data/`, `data_sports/`, `data_toys/`).
2. **Content embeddings** — item text (title, brand, category, price) encoded with a Sentence-T5 model.
3. **RQ-VAE training** — 3-level residual-quantized VAE (encoder `[512, 256, 128]`, latent dim 32, codebook size 256/level) converts each item's embedding into a 3-tuple Semantic ID; a 4th disambiguation token resolves collisions. Trained independently per dataset.
4. **Retrieval training** — a Transformer encoder-decoder (`d_model=128`, 6 heads, 4 encoder + 4 decoder layers) is trained to predict the next item's Semantic ID given the user's interaction history, encoded as a sequence of Semantic ID tokens.
5. **Evaluation** — beam-search decoding (beam width 50) + Recall@K / NDCG@K against the held-out test item.

All code runs end-to-end on Kaggle (GPU T4×2); see `tiger_kaggle_workflow.ipynb`.

## Results

| Metric | Beauty (ours / paper) | Sports (ours / paper) | Toys (ours / paper) |
|---|---|---|---|
| Recall@5  | 0.0280 / 0.0454 | 0.0136 / 0.0264 | 0.0147 / 0.0521 |
| NDCG@5    | 0.0185 / 0.0321 | 0.0088 / 0.0181 | 0.0093 / 0.0371 |
| Recall@10 | 0.0419 / 0.0648 | 0.0222 / 0.0400 | 0.0237 / 0.0712 |
| NDCG@10   | 0.0230 / 0.0384 | 0.0116 / 0.0225 | 0.0122 / 0.0432 |

**Relative to paper (ours ÷ paper):**

| Metric | Beauty | Sports | Toys |
|---|---|---|---|
| Recall@5  | 61.7% | 51.5% | 28.2% |
| NDCG@5    | 57.6% | 48.6% | 25.1% |
| Recall@10 | 64.7% | 55.5% | 33.3% |
| NDCG@10   | 59.9% | 51.6% | 28.2% |

Beauty and Sports land in a broadly similar range (~49–65% of the paper's numbers). **Toys is a clear outlier at ~25–33%** — flagged below as an open question, not smoothed over.

### RQ-VAE codebook health

| Dataset | Recon. loss | Utilization | Perplexity (L1/L2/L3) | Unique 3-token IDs | Collisions |
|---|---|---|---|---|---|
| Beauty | 0.867 | 100% / 100% / 100% | 226 / 191 / 193 | 86.8% | 1,595 of 12,101 |
| Sports and Outdoors | 0.881 | 100% / 100% / 100% | 242 / 230 / 227 | 99.6% | 82 of 18,357 |
| Toys and Games | 0.921 | 100% / 100% / 100% | 228 / 207 / 215 | 84.2% | 1,885 of 11,924 |

All three codebooks are healthy (100% utilization, near-max perplexity, no collapse). Sports has a far lower collision rate than Beauty or Toys, consistent with it having the largest, most content-diverse item catalog of the three.

### Training length

Training used the paper's reported hyperparameters (Adagrad lr=0.4 style config for RQ-VAE; Adafactor, peak_lr=0.01, 10K-step warmup + inverse-sqrt decay for the retrieval Transformer) on all three datasets. `total_steps` was configured to **200,000 for Beauty** and **100,000 for Sports and Toys**.

> **Note:** the paper itself specifies 200,000 steps for Sports (same as Beauty) and only reserves the 100,000-step budget for Toys, due to its smaller size. Sports was run with a 100,000-step budget in this reproduction — that is a deviation from the paper on our part, not one the paper calls for. It's flagged here rather than silently matching the "Yes" hyperparameter checklist.

Early stopping (patience = 5 evaluations / 25,000 steps without validation NDCG@10 improvement) triggered before the configured limit on every dataset, indicating convergence rather than a run cut short:

| Dataset | Steps configured | Best val. NDCG@10 (step) | Stopped at | % of budget used |
|---|---|---|---|---|
| Beauty | 200,000 | 0.0341 (step 20,000) | step 55,000 | 27.5% |
| Sports and Outdoors | 100,000 | 0.0159 (step 70,000) | step 95,000 | 95% (near-full budget) |
| Toys and Games | 100,000 | 0.0181 (step 45,000) | step 70,000 | 70% |

Sports came within 5,000 steps of its (non-paper-matching) budget cap before early-stopping triggered — see `val_curve.png` per dataset for the full training curves. The best checkpoint by validation NDCG@10, not the last one, was used for final evaluation in every case.

### Why results don't fully match the paper

Multiple independent reproductions of this paper (including other open-source attempts) consistently land below the paper's reported metrics, even when matching all stated hyperparameters. The paper was trained with Google's internal T5X framework, which has tokenization/training details (exact vocabulary construction, hashing scheme for user tokens, etc.) that are not fully specified in the paper text. This is a known, documented gap in the community — not evidence of a broken pipeline. (Sanity check: `invalid_rate` on generated Semantic IDs stays well under 1% at K=10 on every dataset — Beauty ≈0.02%, Sports ≈0.0006%, Toys ≈0.16% — confirming beam search almost always lands on a valid item, i.e. the model learned the task correctly on all three.)

Two points worth flagging now that all three datasets can be compared:

- **Toys underperforms Beauty and Sports by a wide margin** (~25–33% of paper vs. ~49–65%). Toys does have the highest RQ-VAE reconstruction loss and a relatively high collision rate of the three, but Beauty's collision rate is nearly as high while Beauty reproduces much closer to the paper — so codebook quality alone doesn't fully explain the size of the gap. Open question, not yet resolved.
- **Model parameter count mismatch**: the trained checkpoints for all three datasets report `model_params = 4,847,104` (≈4.85M), not the "around 13 million parameters" the paper states for this exact configuration (d_model 128, 4+4 layers, 6 heads, MLP 1024, vocab ≈3,024 tokens). Identical and reproducible across all three datasets, so it's not a one-off logging bug — it points to a real difference in vocabulary size, embedding tying, or layer construction relative to the paper's implementation, even though every individual hyperparameter checks out. Worth resolving before treating the hyperparameter match as complete.
- **Sports step-budget deviation** (100,000 vs. the paper's 200,000, above): the run early-stopped only 5,000 steps short of that cap, so it's plausible — though unconfirmed without re-running — that Sports could improve further on the paper's full 200,000-step budget. Unlike the Toys gap, this one is directly testable by re-running with the correct budget.

## Reproducing

```bash
pip install -r requirements.txt

# Beauty
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

# Sports and Outdoors — same stages, sports config + data_sports/ paths
python -m tiger.scripts.preprocess_beauty --category sports --download
python -m tiger.rqvae.train --config configs/rqvae/sports.yaml
python -m tiger.scripts.build_sid_tables \
    --checkpoint outputs/amazon_sports_checkpoints/best.pt \
    --items-csv  outputs/amazon_sports_items.csv \
    --embeddings outputs/amazon_sports_embeddings.npy \
    --sequences-dir data_sports/processed \
    --output-dir   data_sports
python -m tiger.retrieval.train --config configs/retrieval/sports.yaml
python -m tiger.retrieval.evaluate --checkpoint outputs/tiger_sports/checkpoints/best.pt

# Toys and Games — same stages, toys config + data_toys/ paths
python -m tiger.scripts.preprocess_beauty --category toys --download
python -m tiger.rqvae.train --config configs/rqvae/toys.yaml
python -m tiger.scripts.build_sid_tables \
    --checkpoint outputs/amazon_toys_checkpoints/best.pt \
    --items-csv  outputs/amazon_toys_items.csv \
    --embeddings outputs/amazon_toys_embeddings.npy \
    --sequences-dir data_toys/processed \
    --output-dir   data_toys
python -m tiger.retrieval.train --config configs/retrieval/toys.yaml
python -m tiger.retrieval.evaluate --checkpoint outputs/tiger_toys/checkpoints/best.pt
```

Or run `tiger_kaggle_workflow.ipynb` on Kaggle (GPU + Internet enabled) for the full end-to-end pipeline with sanity checks at each stage — the same notebook was used for all three datasets, swapping only the dataset config.

**Note on the embedding model:** the config uses `sentence-t5-base` rather than `sentence-t5-xxl`; the XXL variant (~5B params) does not fit in a 16GB GPU at encoding time. Both produce 768-dim embeddings, so this only affects embedding quality, not the pipeline architecture. Applies to all three datasets.

## Code structure (previous → current)

An earlier version of this project used a flat-file layout, built and validated on Beauty only. The mapping below helps locate any file from that version in the current, package-based structure — no pipeline stage was removed, only reorganized and then reused as-is for Sports and Toys:

| Previous (file / folder) | Current equivalent |
|---|---|
| `train_rqvae.py` | `tiger/rqvae/train.py` |
| `train_decoder.py`, `train_decoder_1.py` | `tiger/retrieval/train.py` |
| `evaluate/` folder, `test_metrics.py`, `Evaluation Metric Testing.ipynb` | `tiger/retrieval/evaluate.py`, `tiger/retrieval/eval.py` |
| `model.py` | `tiger/rqvae/model.py` + `tiger/retrieval/model.py` (two separate models — RQ-VAE and Transformer) |
| `modules/` folder (encoder, quantize, kmeans, loss, etc.) | `tiger/rqvae/encoder.py`, `tiger/rqvae/quantizer.py`, `tiger/rqvae/metrics.py` |
| `data/` folder (`amazon.py`, `preprocessing.py`) | `tiger/scripts/preprocess_beauty.py`, `tiger/rqvae/dataloader_beauty.py` (reused for all 3 datasets) |
| `requirements.txt` | `requirements.txt` (same) |
| (Semantic ID generation, done separately) | `tiger/scripts/build_sid_tables.py`, `tiger/rqvae/generate_sids.py` |
| — (new for this submission) | `data_sports/`, `data_toys/` — per-dataset processed sequences and Semantic ID lookup tables |

## Acknowledgements

This implementation is built on top of an existing open-source reproduction of the TIGER paper, used and extended as a base for this project (hyperparameter correction, full Kaggle training pipeline, debugging, and evaluation against the paper's reported numbers on all three of the paper's benchmark datasets).

Paper: Rajput, S., Mehta, N., Singh, A., et al. *"Recommender Systems with Generative Retrieval."* NeurIPS 2023.
