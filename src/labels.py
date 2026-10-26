"""Milestone 5: failure labels. The last GPU step.

Settings: n in {25, 50, 100} construction pairs x 3 seeds, plus n = 400 once (10 settings).
For each: pick n pairs (stratified: half syc=A, half syc=B), build the balanced v̄ at LAYER,
steer all evaluation prompts with the fixed ALPHA, record Δm and failure = Δm < 0.

Run:  python -m src.labels --model Qwen/Qwen2.5-1.5B-Instruct
Saves results/labels.npz (milestone 6 needs the chosen indices to bootstrap from the same pairs)
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from src.config import ALPHA, DEBUG_MODEL, LAYER, RESULTS, STEER_FROM
from src.data import load_split
from src.model import load_acts, load_model, load_tokenizer
from src.steer import steered_run, steering_vector

NS = [25, 50, 100]
SEEDS = [0, 1, 2]


def construct_letters(acts):
    """syc_is_a for each construction pair, in the same order as acts["diffs"]."""
    construct, _ = load_split()
    by_id = {it["id"]: it for it in construct}
    return torch.tensor([by_id[i]["syc_letter"] == "A" for i in acts["construct_ids"]])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only the first N evaluation prompts (debug)")
    parser.add_argument("--ns", default=None, help="comma-separated n values (debug), default 25,50,100 + all")
    args = parser.parse_args()

    acts = load_acts(args.model)
    syc_is_a = construct_letters(acts)
    n_all = len(syc_is_a)
    ns = [int(x) for x in args.ns.split(",")] if args.ns else NS
    settings = [(n, s) for n in ns for s in SEEDS] + [(n_all, 0)]

    _, evaluate = load_split()
    k = min(args.limit or len(acts["m_base"]), len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    eval_a = np.array([it["syc_letter"] == "A" for it in items])
    tok, model = load_tokenizer(args.model), load_model(args.model)
    print(f"layer {LAYER}, alpha {ALPHA}, steer from {STEER_FROM}, {k} eval prompts, {len(settings)} settings")

    dm = np.zeros((len(settings), k), dtype=np.float32)
    mass = np.zeros_like(dm)
    idx_pad = -np.ones((len(settings), n_all), dtype=np.int64)  # chosen pairs, padded with -1
    for i, (n, seed) in enumerate(settings):
        idx = sample_pairs(syc_is_a, n, seed)
        idx_pad[i, : len(idx)] = idx
        v = steering_vector(acts["diffs"], LAYER, idx, syc_is_a)
        out = [steered_run(model, tok, it, v, ALPHA, LAYER, STEER_FROM)
               for it in tqdm(items, desc=f"n={n} seed={seed}", leave=False)]
        dm[i] = np.array([m for m, _ in out]) - m_base.numpy()
        mass[i] = [p for _, p in out]
        fail = dm[i] < 0
        print(f"n={n:3d} seed={seed}  |v|={v.norm():.2f}  mean Δm={dm[i].mean():+.2f}  fail={fail.mean():.2f}"
              f" (syc=A {fail[eval_a].mean():.2f}, syc=B {fail[~eval_a].mean():.2f})  mass={mass[i].mean():.2f}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "labels.npz"
    np.savez(path, n=np.array([n for n, _ in settings]), seed=np.array([s for _, s in settings]),
             idx=idx_pad, dm=dm, fail=dm < 0, mass=mass, eval_ids=np.array(acts["eval_ids"][:k]),
             m_base=m_base.numpy(), layer=LAYER, alpha=ALPHA, steer_from=STEER_FROM, model=args.model)
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
