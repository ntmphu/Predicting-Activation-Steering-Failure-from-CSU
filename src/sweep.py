"""Milestone 4: choose the steering layer and alpha for one (model, behavior) run.

For each layer and strength, steer N evaluation prompts with the letter-balanced v̄ (all construction pairs)
and record mean Δm, failure share (Δm < 0), and answer mass P("(A")+P("(B") (below ~0.5 = model breaking).

Strength is set relative to the hidden-state size so one grid works for every layer and model:
    r = alpha * |v̄| / mean ||h||   ->   alpha = r * mean ||h|| / |v̄|
Layers are picked by relative depth, so the same grid fits 28-layer (1.5B) and 36-layer (3B) models.

The best setting (mass > 0.5, 5% <= fail <= 40%, largest mean Δm) is saved to <run_dir>/setup.json,
which milestone 5 reads.

Run:  python -m src.sweep --model Qwen/Qwen2.5-1.5B-Instruct --behavior sycophancy --limit 100
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # save figures without a display (Colab / terminal)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.config import BEHAVIOR, DEBUG_MODEL, STEER_FROM, run_dir, save_setup
from src.data import load_split
from src.model import construct_letters, load_acts, load_model, load_tokenizer
from src.steer import steered_run, steering_vector

DEPTHS = [0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75]  # layer = round(depth * n_layers)
STRENGTHS = [0.05, 0.1, 0.25, 0.5]                # r: push size as a fraction of ||h||


def plot(df, mode, path, title):
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
    for ax, t in zip(axes, ["mean Δm (want > 0)", "failure share Δm < 0 (want 5–40%)",
                            "P('(A')+P('(B') (want > 0.5)"]):
        ax.set_title(t); ax.set_xlabel("layer")
    axes[0].legend()
    fig.suptitle(f"{title}, steer from = {mode}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--behavior", default=BEHAVIOR)
    parser.add_argument("--limit", type=int, default=100, help="evaluation prompts per setting")
    parser.add_argument("--modes", default=STEER_FROM, help="comma-separated: all,reply")
    args = parser.parse_args()

    acts = load_acts(args.model, args.behavior)
    syc_is_a = construct_letters(acts)
    _, evaluate = load_split(args.behavior)
    k = min(args.limit, len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    tok, model = load_tokenizer(args.model), load_model(args.model)
    n_layers = model.config.num_hidden_layers
    layers = sorted({round(d * n_layers) for d in DEPTHS})
    modes = args.modes.split(",")

    rows = []
    for mode in modes:
        for layer in layers:
            v = steering_vector(acts["diffs"], layer, syc_is_a=syc_is_a)
            h_norm = acts["eval_h"][:, layer].norm(dim=-1).mean().item()
            for r in STRENGTHS:
                alpha = r * h_norm / v.norm().item()
                out = [steered_run(model, tok, it, v, alpha, layer, mode)
                       for it in tqdm(items, desc=f"{mode} L{layer} r={r}", leave=False)]
                dm = torch.tensor([m for m, _ in out]) - m_base
                mass = torch.tensor([p for _, p in out])
                rows.append({"mode": mode, "layer": layer, "r": r, "alpha": round(alpha, 3),
                             "v_norm": round(v.norm().item(), 3), "h_norm": round(h_norm, 1),
                             "mean_dm": dm.mean().item(), "fail": (dm < 0).float().mean().item(),
                             "mass": mass.mean().item()})
                print(f"{mode:5s} L{layer:2d} r={r:<4} alpha={alpha:7.2f}  mean Δm={dm.mean():+6.2f}"
                      f"  fail={(dm < 0).float().mean():.2f}  mass={mass.mean():.2f}")

    out_dir = run_dir(args.model, args.behavior)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep.csv", index=False)
    for mode in modes:
        plot(df, mode, out_dir / "figures" / f"sweep_{mode}.png", f"{args.model.split('/')[-1]}, {args.behavior}")

    # Pick: model not broken, failure rate in range, biggest average push
    ok = df[(df["mass"] > 0.5) & (df["fail"].between(0.05, 0.40)) & (df["mean_dm"] > 0)]
    print("\nCandidates (mass > 0.5, 5% <= fail <= 40%, mean Δm > 0), best first:")
    print(ok.sort_values("mean_dm", ascending=False).head(8).to_string(index=False) if len(ok) else "  none")
    if len(ok) == 0:
        ok = df[(df["mass"] > 0.5) & (df["mean_dm"] > 0)]
        print("WARNING: no setting in the 5-40% failure range; falling back to the largest positive mean Δm")
    assert len(ok), "steering never worked (mean Δm <= 0 everywhere); cannot choose a setup"
    best = ok.sort_values("mean_dm", ascending=False).iloc[0]
    setup = {"layer": int(best["layer"]), "alpha": float(best["alpha"]), "r": float(best["r"]),
             "steer_from": best["mode"], "mean_dm": float(best["mean_dm"]), "fail": float(best["fail"]),
             "mass": float(best["mass"]), "chosen_by": "src.sweep (balanced v̄)"}
    save_setup(setup, args.model, args.behavior)
    print(f"\nchosen setup -> {out_dir / 'setup.json'}: {setup}")


if __name__ == "__main__":
    main()
