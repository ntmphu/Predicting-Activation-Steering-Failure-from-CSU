"""Milestone 5: failure labels. The last GPU step.

Settings: each n in --ns x --seeds random draws, plus all construction pairs once (the reference vector).
For each: pick n pairs (stratified: half syc=A, half syc=B), build the balanced v̄ at the chosen layer,
steer all evaluation prompts with the FIXED alpha from setup.json, record Δm and failure = Δm < 0.

Run:  python -m src.labels --model Qwen/Qwen2.5-1.5B-Instruct --behavior sycophancy
      python -m src.labels ... --ns 10,16,25,50 --seeds 8 --tag e2     (E2: many draws at small n)
Saves <run_dir>/labels[_tag].npz (milestone 6 needs the chosen indices to bootstrap from the same pairs)
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.config import BEHAVIOR, DEBUG_MODEL, load_setup, run_dir
from src.data import load_split
from src.model import construct_letters, load_acts, load_model, load_tokenizer
from src.steer import steered_run, steering_vector


def sample_pairs(syc_is_a, n, seed):
    """n pair indices, stratified by letter (n//2 A-pairs, the rest B-pairs), without replacement.
    n >= number of pairs -> all pairs."""
    if n >= len(syc_is_a):
        return np.arange(len(syc_is_a))
    rng = np.random.default_rng([n, seed])  # different, reproducible draw per (n, seed)
    a_idx, b_idx = np.flatnonzero(syc_is_a.numpy()), np.flatnonzero(~syc_is_a.numpy())
    n_a = n // 2
    return np.sort(np.concatenate([rng.choice(a_idx, n_a, replace=False),
                                   rng.choice(b_idx, n - n_a, replace=False)]))


def labels_path(model, behavior, tag=""):
    return run_dir(model, behavior) / f"labels{'_' + tag if tag else ''}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--behavior", default=BEHAVIOR)
    parser.add_argument("--limit", type=int, default=None, help="only the first N evaluation prompts (debug)")
    parser.add_argument("--ns", default="25,50,100", help="comma-separated construction-set sizes")
    parser.add_argument("--seeds", type=int, default=3, help="random draws per n")
    parser.add_argument("--tag", default="", help="suffix for the output file, e.g. e2")
    args = parser.parse_args()

    setup = load_setup(args.model, args.behavior)
    layer, alpha, steer_from = setup["layer"], setup["alpha"], setup["steer_from"]
    acts = load_acts(args.model, args.behavior)
    syc_is_a = construct_letters(acts)
    n_all = len(syc_is_a)
    settings = [(int(n), s) for n in args.ns.split(",") for s in range(args.seeds)] + [(n_all, 0)]

    _, evaluate = load_split(args.behavior)
    k = min(args.limit or len(acts["m_base"]), len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    eval_a = np.array([it["syc_letter"] == "A" for it in items])
    tok, model = load_tokenizer(args.model), load_model(args.model)
    print(f"{args.model} / {args.behavior}: layer {layer}, alpha {alpha}, steer from {steer_from}, "
          f"{k} eval prompts, {len(settings)} settings")

    dm = np.zeros((len(settings), k), dtype=np.float32)
    mass = np.zeros_like(dm)
    idx_pad = -np.ones((len(settings), n_all), dtype=np.int64)  # chosen pairs, padded with -1
    for i, (n, seed) in enumerate(settings):
        idx = sample_pairs(syc_is_a, n, seed)
        idx_pad[i, : len(idx)] = idx
        v = steering_vector(acts["diffs"], layer, idx, syc_is_a)
        out = [steered_run(model, tok, it, v, alpha, layer, steer_from)
               for it in tqdm(items, desc=f"n={n} seed={seed}", leave=False)]
        dm[i] = np.array([m for m, _ in out]) - m_base.numpy()
        mass[i] = [p for _, p in out]
        fail = dm[i] < 0
        print(f"n={n:3d} seed={seed}  |v|={v.norm():.2f}  mean Δm={dm[i].mean():+.2f}  fail={fail.mean():.2f}"
              f" (syc=A {fail[eval_a].mean():.2f}, syc=B {fail[~eval_a].mean():.2f})  mass={mass[i].mean():.2f}")

    path = labels_path(args.model, args.behavior, args.tag)
    np.savez(path, n=np.array([n for n, _ in settings]), seed=np.array([s for _, s in settings]),
             idx=idx_pad, dm=dm, fail=dm < 0, mass=mass, eval_ids=np.array(acts["eval_ids"][:k]),
             m_base=m_base.numpy(), layer=layer, alpha=alpha, steer_from=steer_from, model=args.model,
             behavior=args.behavior)
    print(f"saved {path}")

    # Done-when: failure rate per n (smaller n should fail more); fallback if failures are rare
    print("\nfailure rate per n:")
    ns_arr = np.array([n for n, _ in settings])
    for n in sorted(set(ns_arr)):
        print(f"  n={n:3d}: {(dm[ns_arr == n] < 0).mean():.3f}")
    if (dm < 0).mean() < 0.03:
        print("WARNING: failures < 3% -> use the fallback label 'Δm in the bottom 20%' (coding plan M5)")


if __name__ == "__main__":
    main()
