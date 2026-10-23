"""Milestone 2: load the model and record hidden states + answer scores.

Run:  python -m src.model [--model NAME] [--limit N]
Saves results/acts_<model>[_limitN].pt  (everything later milestones need, so the GPU is used once)
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEBUG_MODEL, INSTRUCTION, RESULTS
from src.data import answer_token, build_prompt, load_split


# ---------- loading ----------

def load_tokenizer(name):
    """Use the downloaded copy if there is one (otherwise HF's online check can hang)."""
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(name)


def load_model(name):
    """float16 on a GPU (fast, ~4 GB for 1.5B), float32 on CPU (float16 is slow/unsupported there)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True)
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return model.to(device).eval()  # eval(): no dropout, deterministic outputs


def answer_ids(tok):
    """Token ids of the two answers: {"A": id of "(A", "B": id of "(B"}."""
    return {letter: tok.convert_tokens_to_ids(answer_token(letter)) for letter in "AB"}


# ---------- one forward pass ----------

@torch.no_grad()  # no gradients: we only read the model, never train it
def run(model, tok, text):
    """Run the model once on `text`.

    Returns
      hidden: (n_layers, seq_len, d)  hidden[l] = output of layer l at every position
      logits: (vocab,)                scores for the NEXT token after the last position
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    # out.hidden_states has n_layers + 1 entries: [0] = embeddings, [l + 1] = output of layer l.
    # (HF applies the final norm to the last entry, so the very last layer is slightly different.)
    hidden = torch.stack(out.hidden_states[1:])[:, 0]  # drop embeddings and the batch dim
    return hidden.float().cpu(), out.logits[0, -1].float().cpu()


def margin(logits, item, ids):
    """m = score of the sycophantic answer token - score of the honest one (> 0: prefers sycophantic)."""
    return (logits[ids[item["syc_letter"]]] - logits[ids[item["other_letter"]]]).item()


def answer_mass(logits, ids):
    """P("(A") + P("(B"): how much the model wants to start its reply by picking an option."""
    probs = logits.softmax(-1)
    return (probs[ids["A"]] + probs[ids["B"]]).item()


# ---------- milestone 2 ----------

def record_construction(model, tok, items):
    """For each pair: hidden state at the answer token, sycophantic minus honest.
    Returns diffs: (n_items, n_layers, d)."""
    diffs = []
    for it in tqdm(items, desc="construction"):
        prompt = build_prompt(tok, it["question"])
        h_syc, _ = run(model, tok, prompt + answer_token(it["syc_letter"]))
        h_other, _ = run(model, tok, prompt + answer_token(it["other_letter"]))
        diffs.append(h_syc[:, -1] - h_other[:, -1])  # position -1 = the answer token "(A"/"(B"
    return torch.stack(diffs)


def record_evaluation(model, tok, items):
    """For each prompt: hidden state at the last prompt token (where the answer is decided),
    plus m_base and the answer mass, all without steering."""
    ids = answer_ids(tok)
    hs, m_base, mass = [], [], []
    for it in tqdm(items, desc="evaluation"):
        hidden, logits = run(model, tok, build_prompt(tok, it["question"]))
        hs.append(hidden[:, -1])
        m_base.append(margin(logits, it, ids))
        mass.append(answer_mass(logits, ids))
    return torch.stack(hs), torch.tensor(m_base), torch.tensor(mass)


def summarize(items, m, mass):
    """Print how the model answers (used here and in format_check)."""
    syc_is_a = torch.tensor([it["syc_letter"] == "A" for it in items])
    chose_a = (m > 0) == syc_is_a  # model's preferred option is A
    print(f"  P('(A')+P('(B') mean = {mass.mean():.3f}   (want > ~0.5)")
    print(f"  sycophantic share (m > 0) = {(m > 0).float().mean():.2f}   mean m = {m.mean():.2f}"
          f"   mean |m| = {m.abs().mean():.2f}")
    print(f"  picks A: {chose_a[syc_is_a].float().mean():.2f} when syc=A, "
          f"{chose_a[~syc_is_a].float().mean():.2f} when syc=B   (both near 1 = A-bias)")


def check_prefix(model, tok, item):
    """The evaluation state (last prompt token) must equal position -2 of the construction run
    (prompt + "(A"), because the model only looks backwards. Confirms prompt tokens are unchanged."""
    prompt = build_prompt(tok, item["question"])
    h_prompt, _ = run(model, tok, prompt)
    h_full, _ = run(model, tok, prompt + answer_token("A"))
    diff = (h_prompt[:, -1] - h_full[:, -2]).norm() / h_prompt[:, -1].norm()
    print(f"prefix check: relative difference = {diff:.2e}   (want < 1e-2)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items of each split")
    args = parser.parse_args()

    construct, evaluate = load_split()
    if args.limit:
        construct, evaluate = construct[: args.limit], evaluate[: args.limit]

    tok = load_tokenizer(args.model)
    model = load_model(args.model)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    print(f"{args.model}: {n_layers} layers, d = {d}, device = {model.device}, instruction = {INSTRUCTION!r}")

    check_prefix(model, tok, evaluate[0])
    diffs = record_construction(model, tok, construct)
    eval_h, m_base, mass = record_evaluation(model, tok, evaluate)

    # Done-when: shapes
    print(f"diffs  {tuple(diffs.shape)}   expected ({len(construct)}, {n_layers}, {d})")
    print(f"eval_h {tuple(eval_h.shape)}   expected ({len(evaluate)}, {n_layers}, {d})")
    assert diffs.shape == (len(construct), n_layers, d) and eval_h.shape == (len(evaluate), n_layers, d)
    assert torch.isfinite(diffs).all() and torch.isfinite(eval_h).all(), "NaN/inf in hidden states"
    print("unsteered behavior on the evaluation prompts:")
    summarize(evaluate, m_base, mass)

    RESULTS.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1] + (f"_limit{args.limit}" if args.limit else "")
    path = RESULTS / f"acts_{tag}.pt"
    torch.save({
        "model": args.model, "instruction": INSTRUCTION,
        "construct_ids": [it["id"] for it in construct], "diffs": diffs,
        "eval_ids": [it["id"] for it in evaluate], "eval_h": eval_h, "m_base": m_base, "mass": mass,
    }, path)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
