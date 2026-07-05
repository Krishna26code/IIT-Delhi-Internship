"""
Sanity-check for evaluate/metrics.py's TopKAccumulator.

Run this from inside the Recommender_System repo folder on Kaggle:
    python test_metrics.py

It builds a tiny, hand-crafted example where the correct Recall@K / NDCG@K
values are known in advance (worked out by hand below), then checks that
TopKAccumulator produces exactly those values. If it prints PASSED, the
metric implementation is correct.
"""
import math
import torch
from evaluate.metrics import TopKAccumulator

# ---- Hand-crafted example ----
# 4 users. Ground-truth Semantic ID is a 2-tuple per user.
# top_k has shape (B, K, D): K=5 ranked candidates per user (best=index 0).
actual = torch.tensor([
    [1, 1],   # user 0: true item = (1,1)
    [2, 2],   # user 1: true item = (2,2)
    [3, 3],   # user 2: true item = (3,3)
    [4, 4],   # user 3: true item = (4,4)
])

top_k = torch.tensor([
    [[1,1],[9,9],[9,9],[9,9],[9,9]],   # user 0: hit at rank 0 (best possible)
    [[9,9],[9,9],[2,2],[9,9],[9,9]],   # user 1: hit at rank 2
    [[9,9],[9,9],[9,9],[9,9],[9,9]],   # user 2: miss (not in top-5 at all)
    [[9,9],[9,9],[9,9],[9,9],[4,4]],   # user 3: hit at rank 4 (last slot)
])

acc = TopKAccumulator(ks=[5])
acc.accumulate(actual, top_k)
result = acc.reduce()

# ---- Hand-computed expected values ----
# Recall@5: hits at ranks 0, 2, miss, 4 -> 3 out of 4 users = 0.75
expected_recall = 3 / 4

# NDCG@5: for each hit, 1/log2(rank+2); miss contributes 0. Average over 4 users.
expected_ndcg = (
    1/math.log2(0 + 2) + 1/math.log2(2 + 2) + 0 + 1/math.log2(4 + 2)
) / 4

print("Computed :", result)
print("Expected : recall@5 =", expected_recall, " ndcg@5 =", expected_ndcg)

assert abs(result["recall@5"] - expected_recall) < 1e-6, "Recall@5 MISMATCH"
assert abs(result["ndcg@5"]   - expected_ndcg)   < 1e-6, "NDCG@5 MISMATCH"

print("\nPASSED: metric implementation matches hand-computed values.")
