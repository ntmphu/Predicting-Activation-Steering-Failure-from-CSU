"""E2: does CSU predict how much a prompt's steering effect changes from one construction set to another?

Uses many random construction sets per n (labels_e2.npz -> features_e2.csv). For each n and prompt:
  sd_dm    = std of Δm across the draws        (the real construction-set variability, measured by steering)
  sd_proj  = std of h(x)ᵀv̄ across the draws   (its linear version: what CSU tries to estimate)
  unstable = the prompt fails for some draws but not for others
CSU (mean over draws) is compared with the baselines by Spearman correlation with sd_dm, and by AUROC for
"unstable". If CSU works as intended it should track sd_proj and sd_dm better than norm / |proj| / conf.

Run:  python -m src.variability --model Qwen/Qwen2.5-1.5B-Instruct --behavior sycophancy --tag e2
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import BEHAVIOR, MODEL, run_dir

SIGNALS = ["csu", "rel_csu", "norm", "abs_proj", "conf"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--behavior", default=BEHAVIOR)
    parser.add_argument("--tag", default="e2")
    args = parser.parse_args()
    out = run_dir(args.model, args.behavior)
    df = pd.read_csv(out / f"features_{args.tag}.csv")
    df = df[df["n"] < df["n"].max()]  # drop the single all-pairs reference setting
    df["abs_proj"] = df["proj"].abs()
    grad = "g_csu" in df.columns  # E5 columns, present if src.grad was run before src.csu
    signals = SIGNALS + (["g_csu", "abs_g_z"] if grad else [])
    if grad:
        df["abs_g_z"] = -df["g_z"].abs()  # minus: a small |z| (effect close to zero) should mean unstable

    rows = []
    for n, g in df.groupby("n"):
        agg = dict(sd_dm=("delta_m", "std"), sd_proj=("proj", "std"), fail_rate=("failure", "mean"),
                   csu=("csu", "mean"), rel_csu=("rel_csu", "mean"), norm=("norm", "first"),
                   abs_proj=("abs_proj", "mean"), conf=("conf", "first"))
        if grad:
            agg.update(sd_gproj=("g_proj", "std"), g_csu=("g_csu", "mean"), abs_g_z=("abs_g_z", "mean"))
        p = g.groupby("prompt_id").agg(**agg)
        unstable = ((p["fail_rate"] > 0) & (p["fail_rate"] < 1)).astype(int)
        row = {"n": n, "draws": g["seed"].nunique(), "prompts": len(p), "unstable_share": unstable.mean(),
               "rho(sd_proj, sd_dm)": spearmanr(p["sd_proj"], p["sd_dm"])[0],
               "rho(csu, sd_proj)": spearmanr(p["csu"], p["sd_proj"])[0]}
        if grad:  # E5: does the gradient version of the projection move with the real Δm?
            row["rho(sd_gproj, sd_dm)"] = spearmanr(p["sd_gproj"], p["sd_dm"])[0]
            row["rho(g_csu, sd_gproj)"] = spearmanr(p["g_csu"], p["sd_gproj"])[0]
        for s in signals:
            row[f"rho({s}, sd_dm)"] = spearmanr(p[s], p["sd_dm"])[0]
        for s in signals:
            row[f"AUROC unstable: {s}"] = roc_auc_score(unstable, p[s]) if unstable.nunique() > 1 else float("nan")
        rows.append(row)

    res = pd.DataFrame(rows).set_index("n").T
    res.to_csv(out / f"variability_{args.tag}.csv")
    print("E2. Does CSU track the real construction-set variability of each prompt's steering effect?")
    print("  rho = Spearman across prompts; 'unstable' = fails for some draws, not others\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(res.to_string())
    print(f"\nsaved {out / f'variability_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
