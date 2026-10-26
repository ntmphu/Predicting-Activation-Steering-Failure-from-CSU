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
INSTRUCTION = ""   # "" = plain CAA question; set to ANSWER_INSTRUCTION if format_check says so

# Experiment settings
SEED = 0
N_CONSTRUCT = 400  # construction pairs; the remaining 600 are evaluation
LAYER = None       # chosen in milestone 4
ALPHA = None       # chosen in milestone 4
