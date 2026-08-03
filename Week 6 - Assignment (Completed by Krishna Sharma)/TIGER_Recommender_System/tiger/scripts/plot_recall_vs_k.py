"""Sweep K and plot Recall@K and Invalid-ID-rate@K for a trained checkpoint.

Mirrors the paper's Figure 6 (invalid IDs vs K) and gives a matching Recall@K
curve alongside it. Beam width is taken as max(configured beam width, K) for
every K so the beam is always wide enough to return K candidates.

Usage:
    python -m tiger.scripts.plot_recall_vs_k \
        --checkpoint outputs/tiger_beauty/checkpoints/best.pt \
        --data-dir data \
        --name beauty \
        --ks 1,5,10,15,20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiger.retrieval.train import evaluate_checkpoint


def sweep(checkpoint: Path, data_dir: Path, split: str, ks: list[int], beam_width: int) -> list[dict]:
    rows = []
    for k in ks:
        bw = max(beam_width, k)
        print(f"[sweep] evaluating K={k} (beam_width={bw})...")
        metrics = evaluate_checkpoint(
            checkpoint, data_dir, split=split, beam_width=bw, top_k=k, ks=(k,)
        )
        row = {"k": k, **metrics}
        rows.append(row)
        print(f"[sweep]   recall@{k}={row.get(f'recall@{k}')}  invalid_rate={row.get('invalid_rate')}")
    return rows


def plot(rows: list[dict], name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [r["k"] for r in rows]
    recalls = [r.get(f"recall@{r['k']}") for r in rows]
    invalids = [r.get("invalid_rate", 0.0) * 100 for r in rows]

    # Recall@K vs K
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, recalls, marker="o", color="tab:blue")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"TIGER ({name}) \u2014 Recall@K vs K")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    recall_path = out_dir / f"{name}_recall_vs_k.png"
    fig.savefig(recall_path, dpi=120)
    plt.close(fig)
    print(f"[sweep] wrote {recall_path}")

    # Invalid-ID-rate (%) vs K
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, invalids, marker="x", color="tab:orange")
    ax.set_xlabel("K")
    ax.set_ylabel("Invalid IDs (%)")
    ax.set_title(f"TIGER ({name}) \u2014 Invalid IDs (%) vs K")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    invalid_path = out_dir / f"{name}_invalid_vs_k.png"
    fig.savefig(invalid_path, dpi=120)
    plt.close(fig)
    print(f"[sweep] wrote {invalid_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True, help="dataset label used in filenames/titles, e.g. beauty")
    ap.add_argument("--ks", default="1,5,10,15,20", help="comma-separated list of K values")
    ap.add_argument("--beam-width", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    rows = sweep(args.checkpoint, args.data_dir, args.split, ks, args.beam_width)
    plot(rows, args.name, args.output_dir)

    summary_path = args.output_dir / f"{args.name}_recall_invalid_vs_k.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[sweep] wrote {summary_path}")


if __name__ == "__main__":
    main()
