"""Milestone 6: CSU and baselines for every (setting, evaluation prompt). No GPU needed.

For each setting from milestone 5 (its n chosen pairs):
  - stratified bootstrap: resample the A-pairs and B-pairs separately (with replacement), B times,
    and build the balanced vector v_b each time  -> V: (B, d)
  - H = evaluation hidden states at LAYER (600, d);  P = H @ V.T;  csu = Var_b(P)   (proposal Eq. 4)
  - baselines: proj = H @ v̄, norm = ||h||, conf = |m_base|, syc_is_a; rel_csu = sqrt(csu) / (|proj| + 1e-6)
  - global context: mean cosine between the bootstrap vectors and v̄ (one number per setting)
H1 check: CSU computed with B in {5, 10, 20, 50} vs B = 100 (Spearman); B = 50 is fine if > 0.9.

Run:  python -m src.csu --model Qwen/Qwen2.5-1.5B-Instruct --behavior sycophancy [--tag e2]
Saves <run_dir>/features[_tag].csv (one row per setting x prompt)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import BEHAVIOR, DEBUG_MODEL, run_dir
from src.data import load_split
from src.labels import labels_path
from src.model import construct_letters, load_acts

B = 50
B_CHECK = [5, 10, 20, 50, 100]


def balanced_mean(D, is_a):
    """½ mean of A-pair diffs + ½ mean of B-pair diffs (cancels the letter part)."""
    return 0.5 * D[is_a].mean(0) + 0.5 * D[~is_a].mean(0)


def bootstrap_vectors(D, is_a, n_boot, rng):
    """n_boot balanced vectors, each from a stratified resample (A and B pairs separately, with replacement)."""
    a, b = np.flatnonzero(is_a), np.flatnonzero(~is_a)
    V = np.empty((n_boot, D.shape[1]), dtype=np.float64)
    for i in range(n_boot):
        V[i] = 0.5 * D[rng.choice(a, len(a))].mean(0) + 0.5 * D[rng.choice(b, len(b))].mean(0)
    return V


def csu_of(H, V):
    """CSU(x) = variance over bootstrap vectors of the projected effect h(x)ᵀ v_b."""
    return (H @ V.T).var(axis=1, ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--behavior", default=BEHAVIOR)
    parser.add_argument("--tag", default="", help="read labels_<tag>.npz, write features_<tag>.csv")
    args = parser.parse_args()

    acts = load_acts(args.model, args.behavior)
    lab = np.load(labels_path(args.model, args.behavior, args.tag))
    assert str(lab["model"]) == args.model, f"labels file is for {lab['model']}, not {args.model}"
    layer = int(lab["layer"])

    D_all = acts["diffs"][:, layer].double().numpy()    # (n_construct, d) construction diffs
    is_a_all = construct_letters(acts).numpy()
    k = lab["dm"].shape[1]
    H = acts["eval_h"][:k, layer].double().numpy()      # (n_eval, d) evaluation hidden states
    m_base = lab["m_base"]
    norm = np.linalg.norm(H, axis=1)
    by_id = {it["id"]: it for it in load_split(args.behavior)[1]}
    eval_is_a = np.array([by_id[int(i)]["syc_letter"] == "A" for i in lab["eval_ids"]])

    rows, stability = [], []
    for s, (n, seed) in enumerate(zip(lab["n"], lab["seed"])):
        idx = lab["idx"][s][lab["idx"][s] >= 0]
        D, is_a = D_all[idx], is_a_all[idx]
        rng = np.random.default_rng([int(n), int(seed), 7])
        v_bar = balanced_mean(D, is_a)
        V = bootstrap_vectors(D, is_a, B, rng)
        csu = csu_of(H, V)
        proj = H @ v_bar
        cos = (V @ v_bar) / (np.linalg.norm(V, axis=1) * np.linalg.norm(v_bar))

        # H1: how stable is the CSU ranking for smaller B? (compare to B = 100)
        ref = csu_of(H, bootstrap_vectors(D, is_a, 100, rng))
        rho = {b: spearmanr(csu_of(H, bootstrap_vectors(D, is_a, b, rng)), ref)[0] for b in B_CHECK[:-1]}
        stability.append({"n": int(n), "seed": int(seed), **{f"B={b}": r for b, r in rho.items()}})

        rows.append(pd.DataFrame({
            "n": int(n), "seed": int(seed), "setting": s, "prompt_id": lab["eval_ids"],
            "syc_is_a": eval_is_a.astype(int), "failure": lab["fail"][s].astype(int), "delta_m": lab["dm"][s],
            "mass": lab["mass"][s], "proj": proj, "norm": norm, "conf": np.abs(m_base),
            "csu": csu, "rel_csu": np.sqrt(csu) / (np.abs(proj) + 1e-6), "global_cos": cos.mean(),
        }))
        print(f"n={n:3d} seed={seed}  mean CSU={csu.mean():.3g}  global cos(v_b, v̄)={cos.mean():.3f}"
              f"  Spearman vs B=100: " + "  ".join(f"B={b}:{r:.2f}" for b, r in rho.items()))

    df = pd.concat(rows, ignore_index=True)
    path = run_dir(args.model, args.behavior) / f"features{'_' + args.tag if args.tag else ''}.csv"
    df.to_csv(path, index=False)

    # Done-when summaries
    st = pd.DataFrame(stability)
    print("\nH1 stability (mean Spearman with B=100 over settings):")
    print("  " + "  ".join(f"{c}: {st[c].mean():.3f}" for c in st.columns if c.startswith("B=")))
    print("\nCSU should shrink as n grows (more pairs = steadier vector):")
    print(df.groupby("n")[["csu", "global_cos", "failure"]].mean().to_string())
    print(f"\nsaved {path}  ({len(df)} rows, columns: {', '.join(df.columns)})")


if __name__ == "__main__":
    main()
