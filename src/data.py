"""Milestone 1: load the CAA sycophancy questions, split them, and build prompts.

Run:  python -m src.data [--model NAME] [--limit N]
"""
import argparse
import json
import random

from src.config import DATA_PATH, DEBUG_MODEL, N_CONSTRUCT, SEED, SPLIT_PATH


def load_items(path=DATA_PATH):
    """Read the 1,000 questions and keep just the answer letters ("(A)" -> "A")."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for i, r in enumerate(raw):
        syc = r["answer_matching_behavior"].strip("() ")
        other = r["answer_not_matching_behavior"].strip("() ")
        assert {syc, other} == {"A", "B"}, f"item {i}: unexpected letters {syc}, {other}"
        items.append({"id": i, "question": r["question"], "syc_letter": syc, "other_letter": other})
    return items


def split_items(items, seed=SEED, n_construct=N_CONSTRUCT):
    """Shuffle with a fixed seed: first n_construct -> construction, rest -> evaluation.
    The ids are saved to split.json so every later script uses the same split."""
    ids = [it["id"] for it in items]
    random.Random(seed).shuffle(ids)
    split = {"seed": seed, "construct": sorted(ids[:n_construct]), "evaluate": sorted(ids[n_construct:])}
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f)
    return _apply_split(items, split)


def load_split(path=SPLIT_PATH):
    """Return (construct, evaluate) item lists from the saved split. Used by milestones 2+."""
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    return _apply_split(load_items(), split)


def _apply_split(items, split):
    by_id = {it["id"]: it for it in items}
    return [by_id[i] for i in split["construct"]], [by_id[i] for i in split["evaluate"]]


def build_prompt(tokenizer, question):
    """Chat-formatted prompt ending right where the assistant's answer starts.

    Qwen's tokenizer merges "(A" into ONE token, so we don't end the prompt with "(".
    Instead the model's next token is the answer token "(A" or "(B" (see answer_token)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def answer_token(letter):
    """The single token the model would emit for an answer: "A" -> "(A"."""
    return "(" + letter


def a_share(items):
    """Fraction of items whose sycophantic answer is A (should be near 0.5)."""
    return sum(it["syc_letter"] == "A" for it in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only check the first N eval prompts")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # only the tokenizer is needed here, not the weights

    items = load_items()
    construct, evaluate = split_items(items)

    # Check 1: counts, no overlap, full coverage
    c_ids, e_ids = {it["id"] for it in construct}, {it["id"] for it in evaluate}
    print(f"construct={len(construct)} evaluate={len(evaluate)}")
    assert not (c_ids & e_ids), "construction and evaluation overlap"
    assert len(c_ids | e_ids) == len(items), "some items are missing from the split"

    # Check 2: sycophantic answer is A about half the time
    for name, part in [("all", items), ("construct", construct), ("evaluate", evaluate)]:
        share = a_share(part)
        flag = "" if 0.4 <= share <= 0.6 else "  <-- skewed, consider balancing"
        print(f"syc=A share ({name}): {share:.3f}{flag}")

    # Check 3: one prompt looks right
    tok = AutoTokenizer.from_pretrained(args.model)
    sample = evaluate[: args.limit] if args.limit else evaluate
    prompt = build_prompt(tok, sample[0]["question"])
    print("\n--- example prompt ---")
    print(prompt)
    print("--- end (tail repr:", repr(prompt[-30:]), ") ---")
    print("syc_letter:", sample[0]["syc_letter"], "other_letter:", sample[0]["other_letter"])

    # Check 4 (needed in milestone 2): "(A" / "(B" must each be ONE token, appended after the
    # prompt's tokens unchanged. Then the prompt's last position is the one that predicts the answer.
    ids = tok.convert_tokens_to_ids([answer_token("A"), answer_token("B")])
    print("\nanswer token ids:", ids, "(must not be the unk id", tok.unk_token_id, ")")
    bad = 0
    for it in sample:
        p = build_prompt(tok, it["question"])
        for letter in "AB":
            if tok.tokenize(p + answer_token(letter)) != tok.tokenize(p) + [answer_token(letter)]:
                bad += 1
    # ascii() because the tokenizer shows "\n" as "Ċ", which the Windows console can't print
    print("last 3 tokens of prompt + '(A':", ascii(tok.tokenize(prompt + answer_token("A"))[-3:]))
    print(f"prompts where the answer is not one clean token: {bad} / {2 * len(sample)}")


if __name__ == "__main__":
    main()
