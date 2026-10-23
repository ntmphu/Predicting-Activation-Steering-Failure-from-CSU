"""Milestone 2 decision: should every question end with "Answer with (A) or (B)."?

For each prompt variant, on the first N evaluation prompts (no steering):
  - format:     P("(A")+P("(B"), how often the model starts its reply by picking an option
  - A-bias:     does it pick A no matter which answer is sycophantic?
  - sycophancy: share of prompts with m > 0
  - agreement:  does the sign of m match the option in the model's actual generated reply?

Run:  python -m src.format_check --model Qwen/Qwen2.5-1.5B-Instruct --limit 100 --gen 30
"""
import argparse
import re

import torch
from tqdm import tqdm

from src.config import ANSWER_INSTRUCTION, DEBUG_MODEL
from src.data import build_prompt, load_split
from src.model import answer_ids, answer_mass, load_model, load_tokenizer, margin, run, summarize

VARIANTS = {"no instruction": "", "with instruction": ANSWER_INSTRUCTION}


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=40):
    """Greedy reply (always the most likely token), so results are reproducible."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def chosen_option(reply):
    """First option mentioned in the reply, e.g. "... I'd say (B) Not a problem" -> "B". None if no option."""
    match = re.search(r"\(([AB])\)|\b([AB])\)", reply)
    return (match.group(1) or match.group(2)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=50, help="evaluation prompts to score")
    parser.add_argument("--gen", type=int, default=10, help="of those, how many to generate replies for")
    args = parser.parse_args()

    _, evaluate = load_split()
    items = evaluate[: args.limit]
    tok, model = load_tokenizer(args.model), load_model(args.model)
    ids = answer_ids(tok)

    for name, instruction in VARIANTS.items():
        m, mass, agree, no_option, examples = [], [], 0, 0, []
        for i, it in enumerate(tqdm(items, desc=name)):
            prompt = build_prompt(tok, it["question"], instruction)
            _, logits = run(model, tok, prompt)
            m.append(margin(logits, it, ids))
            mass.append(answer_mass(logits, ids))
            if i < args.gen:
                reply = generate(model, tok, prompt)
                option = chosen_option(reply)
                m_option = it["syc_letter"] if m[-1] > 0 else it["other_letter"]  # option m prefers
                agree += option == m_option
                no_option += option is None
                if len(examples) < 2:
                    examples.append((m_option, option, reply))

        print(f"\n== {name} ==")
        summarize(items, torch.tensor(m), torch.tensor(mass))
        n_gen = min(args.gen, len(items))
        print(f"  agreement (sign of m vs generated reply): {agree}/{n_gen}   no option in reply: {no_option}")
        for m_option, option, reply in examples:
            print(f"    m prefers {m_option}, reply chose {option}: {reply[:100]!r}")


if __name__ == "__main__":
    main()
