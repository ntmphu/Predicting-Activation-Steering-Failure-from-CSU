"""Milestone 3: the steering hook. Adds alpha * vector to one layer's output while the model runs.

Run:  python -m src.steer [--model NAME] [--limit N] [--layer L] [--alpha A] [--steer-from reply|all]
Checks: alpha = 0 changes nothing; removing the hook restores the model; +alpha raises m, -alpha lowers it.
"""
import argparse

import torch

from src.config import BEHAVIOR, DEBUG_MODEL, STEER_FROM
from src.data import build_prompt, load_split
from src.model import answer_ids, answer_mass, load_acts, load_model, load_tokenizer, margin, run


def steering_vector(diffs, layer, idx=None, syc_is_a=None):
    """v̄ = mean of (h_syc - h_other) over the chosen construction pairs (all if idx is None).

    If syc_is_a (bool per pair) is given, the vector is letter-balanced: ½·mean(A-items) + ½·mean(B-items).
    Each diff contains a large "(A vs (B" token-identity part with opposite signs for A- and B-items;
    equal weighting cancels it exactly, even when the two groups have different sizes."""
    idx = torch.arange(len(diffs)) if idx is None else torch.as_tensor(idx)
    d = diffs[idx, layer]
    if syc_is_a is None:
        return d.mean(0)
    a = torch.as_tensor(syc_is_a)[idx]
    return 0.5 * d[a].mean(0) + 0.5 * d[~a].mean(0)


def reply_start(tok, prompt):
    """Index of the last <|im_start|> token, i.e. where the assistant turn begins."""
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    return len(ids) - 1 - ids[::-1].index(im_start)


def add_steering(model, layer, vector, alpha, start=0):
    """Register a hook on model.model.layers[layer] that adds alpha * vector to its output
    at positions >= start. Returns the handle: call handle.remove() to undo."""
    delta = alpha * vector

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # some versions return a tuple
        h = h.clone()  # don't modify the original tensor in place
        h[:, start:] += delta.to(h.dtype).to(h.device)  # match float16/GPU, avoids dtype errors
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    return model.model.layers[layer].register_forward_hook(hook)


def steered_run(model, tok, item, vector, alpha, layer, steer_from=STEER_FROM):
    """m and answer mass for one prompt with steering on. The hook is always removed afterwards."""
    prompt = build_prompt(tok, item["question"])
    start = reply_start(tok, prompt) if steer_from == "reply" else 0
    handle = add_steering(model, layer, vector, alpha, start)
    try:
        _, logits = run(model, tok, prompt)
    finally:
        handle.remove()
    ids = answer_ids(tok)
    return margin(logits, item, ids), answer_mass(logits, ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=10, help="evaluation prompts to test")
    parser.add_argument("--layer", type=int, default=None, help="default: middle layer")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steer-from", default=STEER_FROM, choices=["reply", "all"])
    parser.add_argument("--behavior", default=BEHAVIOR)
    args = parser.parse_args()

    acts = load_acts(args.model, args.behavior)
    _, evaluate = load_split(args.behavior)
    k = min(args.limit, len(acts["m_base"]))
    items, m_base = evaluate[:k], acts["m_base"][:k]
    assert [it["id"] for it in items] == acts["eval_ids"][:k], "split changed since milestone 2"

    tok, model = load_tokenizer(args.model), load_model(args.model)
    layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
    v = steering_vector(acts["diffs"], layer)
    print(f"layer {layer}, |v̄| = {v.norm():.2f}, steer from = {args.steer_from}, {k} prompts")
    print("reply starts at token", reply_start(tok, build_prompt(tok, items[0]["question"])),
          "of", len(tok(build_prompt(tok, items[0]["question"]), add_special_tokens=False)["input_ids"]))

    def delta_m(alpha):
        out = [steered_run(model, tok, it, v, alpha, layer, args.steer_from) for it in items]
        return torch.tensor([m for m, _ in out]) - m_base, torch.tensor([p for _, p in out])

    # Check 1: alpha = 0 must change nothing
    d0, _ = delta_m(0.0)
    print(f"alpha = 0:  max |Δm| = {d0.abs().max():.1e}   (want < 1e-3)")

    # Check 2: after the hook is removed, the plain model gives m_base again
    ids = answer_ids(tok)
    m_again = torch.tensor([margin(run(model, tok, build_prompt(tok, it["question"]))[1], it, ids) for it in items])
    print(f"hook removed: max |m - m_base| = {(m_again - m_base).abs().max():.1e}   (want < 1e-3)")

    # Check 3: the sign is right. +alpha (towards sycophancy) should raise m on average, -alpha lower it
    for a in (args.alpha, -args.alpha):
        d, mass = delta_m(a)
        print(f"alpha = {a:+.1f}: mean Δm = {d.mean():+.2f}   share Δm < 0 = {(d < 0).float().mean():.2f}"
              f"   P('(A')+P('(B') = {mass.mean():.2f}")


if __name__ == "__main__":
    main()
