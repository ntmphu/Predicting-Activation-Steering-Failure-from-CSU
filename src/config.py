"""Shared settings for every milestone. Change values here, not in the scripts."""
from pathlib import Path

# Paths (relative to the repo root, so they work on the laptop and on Colab)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "generate_dataset.json"
SPLIT_PATH = ROOT / "data" / "split.json"
RESULTS = ROOT / "results"
HF_REPO_NAME = "csu-results"  # private HF dataset repo for backups; owner = whoever's token is used

# Models: debug on the small one (laptop CPU), run the real one on Colab
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEBUG_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Prompt: optional text appended to every question (decided with src/format_check.py)
ANSWER_INSTRUCTION = "\n\nAnswer with (A) or (B)."
INSTRUCTION = ANSWER_INSTRUCTION  # chosen Oct 26 from format_check on 1.5B (see PROGRESS.md log)

# Experiment settings
SEED = 0
N_CONSTRUCT = 400  # construction pairs; the remaining 600 are evaluation
# Steering (chosen Oct 26 from src/sweep.py + src/diagnose.py on 1.5B, see PROGRESS.md log)
STEER_FROM = "all"  # every position; "reply" (CAA-style) was too weak on Qwen-1.5B (Δm <= 0.22)
LAYER = 18
ALPHA = 3.48        # r = 0.1 of mean ||h|| with the 400-pair balanced v̄; kept FIXED for every n
# Vectors are always letter-balanced (½ mean of syc=A pairs + ½ mean of syc=B pairs), and pairs are
# sampled stratified by letter, because each diff has a ~25-norm "(A vs (B" part that must cancel.
