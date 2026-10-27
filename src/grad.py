"""E5: the gradient of the margin with respect to the steering vector, for every evaluation prompt (GPU).

Why: steering adds alpha * v to the layer output, so to first order
    Δm ≈ alpha * g(x)ᵀ v      with  g(x) = ∂m / ∂(added vector) = sum over steered positions of ∂m/∂h_t.
E2 showed that the projection h(x)ᵀv does not track Δm. g(x) says how the model READS the added vector,
so gᵀv̄ is the linear prediction of Δm and Var_b[gᵀv_b] is the CSU that should matter (computed in src.csu).

Trick: add a zero vector `delta` (requires_grad) with the same hook as steering, run forward, backprop m.
delta.grad is exactly sum_t ∂m/∂h_t over the steered positions, in one backward pass per prompt.

Run:  python -m src.grad --model Qwen/Qwen2.5-1.5B-Instruct --behavior sycophancy [--limit 20]
Saves <run_dir>/grads.pt  {"eval_ids", "g": (n_eval, d), "layer", "steer_from"}
"""
import argparse

import torch
from tqdm import tqdm

from src.config import BEHAVIOR, DEBUG_MODEL, load_setup, run_dir
from src.data import build_prompt, load_split
from src.model import answer_ids, load_acts, load_model, load_tokenizer
from src.steer import add_steering, reply_start


def margin_grad(model, tok, item, layer, steer_from, ids):
    """g = ∂m/∂delta where delta is added (like a steering vector) at layer `layer`. Also returns m."""
    prompt = build_prompt(tok, item["question"])
    start = reply_start(tok, prompt) if steer_from == "reply" else 0
    delta = torch.zeros(model.config.hidden_size, device=model.device, requires_grad=True)
    handle = add_steering(model, layer, delta, 1.0, start)  # adds 1.0 * delta = zeros: the output is unchanged
    try:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[0, -1].float()
        m = logits[ids[item["syc_letter"]]] - logits[ids[item["other_letter"]]]
        m.backward()  # fills delta.grad; model weights are frozen, so only delta gets a gradient
    finally:
        handle.remove()
    return delta.grad.float().cpu(), m.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--behavior", default=BEHAVIOR)
    parser.add_argument("--limit", type=int, default=None, help="only the first N evaluation prompts (debug)")
    args = parser.parse_args()

    setup = load_setup(args.model, args.behavior)
    layer, steer_from = setup["layer"], setup["steer_from"]
    acts = load_acts(args.model, args.behavior)
    _, evaluate = load_split(args.behavior)
    k = min(args.limit or len(acts["m_base"]), len(acts["m_base"]))
    items = evaluate[:k]
    assert [it["id"] for it in items] == acts["eval_ids"][:k], "split changed since milestone 2"

    tok, model = load_tokenizer(args.model), load_model(args.model)
    model.requires_grad_(False)  # we only need the gradient w.r.t. delta, not the weights
    ids = answer_ids(tok)
    out = [margin_grad(model, tok, it, layer, steer_from, ids) for it in tqdm(items, desc="gradients")]
    G = torch.stack([g for g, _ in out])
    m = torch.tensor([mm for _, mm in out])

    # Done-when: m recomputed here equals m_base from milestone 2 (same prompt, zero delta)
    print(f"layer {layer}, steer from {steer_from}, {k} prompts, |g| mean = {G.norm(dim=1).mean():.3g}")
    print(f"max |m - m_base| = {(m - acts['m_base'][:k]).abs().max():.2e}   (want < ~0.05 in float16)")
    path = run_dir(args.model, args.behavior) / "grads.pt"
    torch.save({"eval_ids": acts["eval_ids"][:k], "g": G, "layer": layer, "steer_from": steer_from}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
