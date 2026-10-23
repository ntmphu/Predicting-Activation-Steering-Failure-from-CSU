"""Show how one dataset item becomes a prompt and how the tokenizer splits it.

Run:  python scripts/inspect_prompt.py                   # item 0, last 15 tokens
      python scripts/inspect_prompt.py --id 7 --last 40
      python scripts/inspect_prompt.py --text "Answer: (A) yes"   # tokenize any text
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported
from transformers import AutoTokenizer

from src.config import DEBUG_MODEL
from src.data import answer_token, build_prompt, load_items


def load_tokenizer(name):
    """Use the downloaded copy if there is one (avoids hanging on the online check)."""
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(name)


def show_tokens(tok, text, last=None):
    """Print one row per token: position, id, raw token, and the actual text it stands for."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    start = 0 if last is None else max(0, len(ids) - last)
    print(f"{'pos':>5} {'id':>7}  {'raw token':<18} text")
    for pos in range(start, len(ids)):
        raw = tok.convert_ids_to_tokens(ids[pos])
        print(f"{pos:>5} {ids[pos]:>7}  {ascii(raw):<18} {tok.decode([ids[pos]])!r}")
    print(f"({len(ids)} tokens total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, help="dataset item to inspect")
    parser.add_argument("--last", type=int, default=15, help="show only the last N prompt tokens")
    parser.add_argument("--text", default=None, help="tokenize this text instead of a dataset item")
    parser.add_argument("--model", default=DEBUG_MODEL)
    args = parser.parse_args()
    tok = load_tokenizer(args.model)

    if args.text is not None:
        show_tokens(tok, args.text)
        return

    item = load_items()[args.id]
    print("=== 1. INPUT (from the dataset) ===")
    print(item["question"])
    print(f"\nsyc_letter={item['syc_letter']}  other_letter={item['other_letter']}")

    prompt = build_prompt(tok, item["question"])
    print("\n=== 2. OUTPUT of build_prompt (what the model reads) ===")
    print(prompt + "<<END>>")  # <<END>> marks where the prompt stops

    print(f"\n=== 3. TOKENS of the prompt (last {args.last}) ===")
    show_tokens(tok, prompt, args.last)

    print("\n=== 4. ANSWER TOKENS the model chooses between ===")
    for letter in "AB":
        t = answer_token(letter)
        role = "sycophantic" if letter == item["syc_letter"] else "honest"
        print(f"{t!r:>6} -> id {tok.convert_tokens_to_ids(t)}  ({role})")
    same = tok.tokenize(prompt + answer_token("A")) == tok.tokenize(prompt) + [answer_token("A")]
    print("prompt + '(A' = prompt tokens + one '(A' token?", same)


if __name__ == "__main__":
    main()
