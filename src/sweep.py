"""Milestone 4: choose the steering layer and alpha.

For each layer and strength, steer N evaluation prompts towards sycophancy with v̄ (all 400 pairs) and record
mean Δm, failure share (Δm < 0), and answer mass P("(A")+P("(B") (if it drops below ~0.5 the model is breaking).

Strength is set relative to the hidden-state size so one grid works for every layer:
    r = alpha * |v̄| / mean ||h||   ->   alpha = r * mean ||h|| / |v̄|

Run:  python -m src.sweep --model Qwen/Qwen2.5-1.5B-Instruct --limit 100
Saves results/sweep.csv and results/figures/sweep_<mode>.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # save figures without a display (Colab / terminal)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.config import DEBUG_MODEL, RESULTS
from src.data import load_split
from src.model import load_acts, load_model, load_tokenizer
from src.steer import steered_run, steering_vector

LAYERS = [6, 8, 10, 12, 14, 16, 18, 20]
STRENGTHS = [0.05, 0.1, 0.25, 0.5, 1.0]  # r: push size as a fraction of ||h||
MODES = ["reply", "all"]           # main setting + robustness alternative


def plot(df, mode, path):
    """Three panels vs layer, one line per strength: mean Δm, failure share, answer mass."""
    d = df[df["mode"] == mode]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for r, g in d.groupby("r"):
        axes[0].plot(g["layer"], g["mean_dm"], marker="o", label=f"r = {r}")
        axes[1].plot(g["layer"], g["fail"], marker="o")
        axes[2].plot(g["layer"], g["mass"], marker="o")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[1].axhspan(0.05, 0.40, color="green", alpha=0.1)  # target failure range
    axes[2].axhline(0.5, color="red", lw=0.8, ls="--")      # below this the model is breaking
    for ax, title in zip(axes, ["mean Δm (want > 0)", "failure share Δm < 0 (want 5–40%)",
                                "P('(A')+P('(B') (want > 0.5)"]):
        ax.set_title(title); ax.set_xlabel("layer")
    axes[0].legend()
    fig.suptitle(f"Steering sweep, steer from = {mode}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=100, help="evaluation prompts per setting")
    args = parser.parse_args()

    acts = load_acts(args.model)
    _, evaluate = load_split()
    k = min(args.limit, len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    tok, model = load_tokenizer(args.model), load_model(args.model)
    layers = [L for L in LAYERS if L < model.config.num_hidden_layers]

    rows = []
    for mode in MODES:
        for layer in layers:
            v = steering_vector(acts["diffs"], layer)
            h_norm = acts["eval_h"][:, layer].norm(dim=-1).mean().item()
            for r in STRENGTHS:
                alpha = r * h_norm / v.norm().item()
                out = [steered_run(model, tok, it, v, alpha, layer, mode)
                       for it in tqdm(items, desc=f"{mode} L{layer} r={r}", leave=False)]
                dm = torch.tensor([m for m, _ in out]) - m_base
                mass = torch.tensor([p for _, p in out])
                rows.append({"mode": mode, "layer": layer, "r": r, "alpha": round(alpha, 2),
                             "v_norm": round(v.norm().item(), 3), "h_norm": round(h_norm, 1),
                             "mean_dm": dm.mean().item(), "fail": (dm < 0).float().mean().item(),
                             "mass": mass.mean().item()})
                print(f"{mode:5s} L{layer:2d} r={r:<4} alpha={alpha:7.2f}  mean Δm={dm.mean():+6.2f}"
                      f"  fail={(dm < 0).float().mean():.2f}  mass={mass.mean():.2f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True); (RESULTS / "figures").mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sweep.csv", index=False)
    for mode in MODES:
        plot(df, mode, RESULTS / "figures" / f"sweep_{mode}.png")

    # Candidates: model not broken, failure rate in range, biggest average push
    ok = df[(df["mass"] > 0.5) & (df["fail"].between(0.05, 0.40)) & (df["mean_dm"] > 0)]
    print("\nCandidates (mass > 0.5, 5% <= fail <= 40%, mean Δm > 0), best first:")
    print(ok.sort_values("mean_dm", ascending=False).head(8).to_string(index=False) if len(ok) else "  none")
    print(f"\nsaved {RESULTS / 'sweep.csv'} and figures/sweep_*.png")


if __name__ == "__main__":
    main()
