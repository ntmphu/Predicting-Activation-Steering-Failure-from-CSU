"""Milestones 7-8: does CSU predict steering failure, and does it help selective steering? No GPU needed.

M7: single-signal AUROC per n, then cross-validated logistic regression (5-fold GroupKFold by prompt,
    so a prompt is never in both train and test) comparing nested models; Δ AUROC with a 95% bootstrap CI.
M8: risk–coverage: steer only the c% of prompts with the lowest predicted risk, measure their failure rate.

Run:  python -m src.analysis [--n 25]
Saves results/auroc_table.csv, results/risk_coverage.csv, results/figures/risk_coverage_n<N>.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS

BASE = ["proj", "abs_proj", "norm", "conf", "syc_is_a"]
MODELS = {  # nested comparisons from the proposal (section 3.3, step 2)
    "projection": ["proj", "abs_proj"],
    "baselines": BASE,
    "CSU only": ["log_csu"],
    "baselines + CSU": BASE + ["log_csu", "log_rel_csu"],
}
POOLED_EXTRA = ["log_n", "global_cos"]  # construction-set size + global reliability (only vary across settings)
SIGNALS = ["proj", "abs_proj", "norm", "conf", "syc_is_a", "csu", "rel_csu"]
COVERAGES = np.round(np.arange(0.1, 1.01, 0.1), 1)


def add_features(df):
    df = df.copy()
    df["abs_proj"] = df["proj"].abs()
    df["log_csu"] = np.log(df["csu"] + 1e-12)        # CSU is very skewed
    df["log_rel_csu"] = np.log(df["rel_csu"] + 1e-12)
    df["log_n"] = np.log(df["n"])
    return df


def oof_risk(df, cols):
    """Out-of-fold predicted failure probability (each prompt is predicted by a model that never saw it)."""
    risk = np.zeros(len(df))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    for train, test in GroupKFold(n_splits=5).split(df, groups=df["prompt_id"]):
        model.fit(df.iloc[train][cols], df.iloc[train]["failure"])
        risk[test] = model.predict_proba(df.iloc[test][cols])[:, 1]
    return risk


def delta_auc_ci(df, risk_a, risk_b, n_boot=500, seed=0):
    """95% CI of AUROC(b) - AUROC(a), resampling prompts (all rows of a prompt move together)."""
    rng = np.random.default_rng(seed)
    pids = df["prompt_id"].to_numpy()
    groups = {p: np.flatnonzero(pids == p) for p in np.unique(pids)}
    keys, y, out = list(groups), df["failure"].to_numpy(), []
    for _ in range(n_boot):
        rows = np.concatenate([groups[p] for p in rng.choice(keys, len(keys))])
        if y[rows].min() == y[rows].max():
            continue
        out.append(roc_auc_score(y[rows], risk_b[rows]) - roc_auc_score(y[rows], risk_a[rows]))
    return np.percentile(out, [2.5, 97.5])


def risk_coverage(y, risk):
    """Failure rate among the c share of prompts with the LOWEST predicted risk, for each coverage c."""
    order = np.argsort(risk, kind="stable")
    return np.array([y[order[: max(1, int(round(c * len(y))))]].mean() for c in COVERAGES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="construction-set size for the risk–coverage figure")
    args = parser.parse_args()
    df = add_features(pd.read_csv(RESULTS / "features.csv"))

    # ---- M7a: each signal on its own ----
    print("M7a. single-signal AUROC per n  ('-' = signal works in reverse, reported as 1 - AUROC)")
    for n, g in df.groupby("n"):
        parts = []
        for s in SIGNALS:
            auc = roc_auc_score(g["failure"], g[s]) if g[s].nunique() > 1 else 0.5
            parts.append(f"{s}={max(auc, 1 - auc):.3f}{'-' if auc < 0.5 else ''}")
        print(f"  n={n:3d} (fail {g['failure'].mean():.2f}): " + "  ".join(parts))

    # ---- M7b: nested models, cross-validated ----
    rows, risks = [], {}
    groups = [(f"n={n}", g.reset_index(drop=True), MODELS) for n, g in df.groupby("n")]
    pooled = {**MODELS, "global reliability": POOLED_EXTRA,
              "baselines + global": BASE + POOLED_EXTRA,
              "baselines + global + CSU": BASE + POOLED_EXTRA + ["log_csu", "log_rel_csu"]}
    groups.append(("all n pooled", df.reset_index(drop=True), pooled))
    for name, g, models in groups:
        for model_name, cols in models.items():
            r = oof_risk(g, cols)
            risks[(name, model_name)] = r
            rows.append({"set": name, "model": model_name, "AUROC": roc_auc_score(g["failure"], r),
                         "AP": average_precision_score(g["failure"], r), "fail_rate": g["failure"].mean()})
        top, base = ("baselines + global + CSU", "baselines + global") if name == "all n pooled" \
            else ("baselines + CSU", "baselines")
        lo, hi = delta_auc_ci(g, risks[(name, base)], risks[(name, top)])
        rows.append({"set": name, "model": f"Δ AUROC ({top} − {base}) 95% CI",
                     "AUROC": (lo + hi) / 2, "AP": np.nan, "CI_low": lo, "CI_high": hi})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "auroc_table.csv", index=False)
    print("\nM7b. cross-validated AUROC / AP (GroupKFold by prompt)")
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(table.drop(columns=["fail_rate"]).to_string(index=False))

    # ---- M8: risk–coverage ----
    rc_rows = []
    for n, g in df.groupby("n"):
        g = g.reset_index(drop=True)
        y = g["failure"].to_numpy()
        curves = {"random": np.full(len(COVERAGES), y.mean())}
        for m in ["projection", "baselines", "CSU only", "baselines + CSU"]:
            curves[m] = risk_coverage(y, risks[(f"n={n}", m)])
        for m, c in curves.items():
            rc_rows.append({"n": n, "model": m, "AURC": np.trapezoid(c, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]),
                            **{f"c={cv}": v for cv, v in zip(COVERAGES, c)}})
        if n == args.n:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for m, c in curves.items():
                ax.plot(COVERAGES * 100, c * 100, marker="o", ls="--" if m == "random" else "-", label=m)
            ax.set_xlabel("coverage: % of prompts steered (lowest predicted risk first)")
            ax.set_ylabel("failure rate among steered prompts (%)")
            ax.set_title(f"Selective steering, n = {n} construction pairs")
            ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
            (RESULTS / "figures").mkdir(exist_ok=True, parents=True)
            fig.savefig(RESULTS / "figures" / f"risk_coverage_n{n}.png", dpi=150); plt.close(fig)
    rc = pd.DataFrame(rc_rows)
    rc.to_csv(RESULTS / "risk_coverage.csv", index=False)
    print("\nM8. area under risk–coverage curve (lower = better)")
    print(rc.pivot(index="model", columns="n", values="AURC").round(3).to_string())
    print(f"\nsaved auroc_table.csv, risk_coverage.csv, figures/risk_coverage_n{args.n}.png")


if __name__ == "__main__":
    main()
