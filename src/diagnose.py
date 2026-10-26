"""Milestone 4 diagnostic: is the steering vector partly a "say A/B" letter direction?

1. Per layer: cosine between v̄_A (mean diff of syc=A pairs) and v̄_B (syc=B pairs). If the letter part
   dominates, they point in opposite directions (cos < 0); a shared sycophancy direction gives cos > 0.
2. For a few settings from the sweep: plain v̄ vs letter-balanced v̄, with failure rate split by syc letter.
   If failures are mostly one letter, failures are letter effects, not sycophancy effects.

Run:  python -m src.diagnose --model Qwen/Qwen2.5-1.5B-Instruct --limit 100
"""
import argparse

import torch
import torch.nn.functional as F

from src.config import DEBUG_MODEL
from src.data import load_split
from src.model import load_acts, load_model, load_tokenizer
from src.steer import steered_run, steering_vector

# (mode, layer, r) settings to compare, taken from the sweep
SETTINGS = [("all", 18, 0.1), ("all", 20, 0.25), ("all", 16, 0.05), ("all", 18, 0.5),
            ("reply", 16, 0.1), ("reply", 18, 0.1), ("reply", 16, 0.5), ("reply", 12, 0.5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    acts = load_acts(args.model)
    construct, evaluate = load_split()
    construct = construct[: len(acts["construct_ids"])]
    syc_is_a = torch.tensor([it["syc_letter"] == "A" for it in construct])
    diffs = acts["diffs"]
    n_layers = diffs.shape[1]
    print(f"construction: {int(syc_is_a.sum())} syc=A, {int((~syc_is_a).sum())} syc=B")

    print("\n1) cos(v̄_A, v̄_B) and vector sizes per layer")
    for layer in range(4, n_layers, 2):
        va, vb = diffs[syc_is_a, layer].mean(0), diffs[~syc_is_a, layer].mean(0)
        plain, bal = steering_vector(diffs, layer), steering_vector(diffs, layer, syc_is_a=syc_is_a)
        print(f"  L{layer:2d}: cos = {F.cosine_similarity(va, vb, dim=0):+.2f}   |v̄_A| = {va.norm():6.2f}"
              f"   |v̄_B| = {vb.norm():6.2f}   |plain| = {plain.norm():.2f}   |balanced| = {bal.norm():.2f}"
              f"   cos(plain, balanced) = {F.cosine_similarity(plain, bal, dim=0):+.2f}")

    k = min(args.limit, len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    eval_a = torch.tensor([it["syc_letter"] == "A" for it in items])
    tok, model = load_tokenizer(args.model), load_model(args.model)

    print("\n2) plain vs letter-balanced vector")
    for mode, layer, r in SETTINGS:
        if layer >= n_layers:
            continue
        h_norm = acts["eval_h"][:, layer].norm(dim=-1).mean().item()
        for name, v in [("plain", steering_vector(diffs, layer)),
                        ("balanced", steering_vector(diffs, layer, syc_is_a=syc_is_a))]:
            alpha = r * h_norm / v.norm().item()
            out = [steered_run(model, tok, it, v, alpha, layer, mode) for it in items]
            dm = torch.tensor([m for m, _ in out]) - m_base
            fail = dm < 0
            mass = torch.tensor([p for _, p in out]).mean()
            print(f"  {mode:5s} L{layer:2d} r={r:<4} {name:8s} alpha={alpha:7.2f}  mean Δm={dm.mean():+.2f}"
                  f"  fail={fail.float().mean():.2f} (syc=A {fail[eval_a].float().mean():.2f},"
                  f" syc=B {fail[~eval_a].float().mean():.2f})  mass={mass:.2f}")


if __name__ == "__main__":
    main()
