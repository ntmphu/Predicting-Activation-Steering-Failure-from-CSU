"""Shared settings for every milestone. Change values here, not in the scripts."""
import json
from pathlib import Path

# Paths (relative to the repo root, so they work on the laptop and on Colab)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
HF_REPO_NAME = "csu-results"  # private HF dataset repo for backups; owner = whoever's token is used

# Models: debug on the small one (laptop CPU), run the real ones on Colab
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEBUG_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Behaviors = CAA A/B datasets in data/<behavior>.json. "syc_letter" in the code means the letter of the
# answer showing the target behavior (answer_matching_behavior), whatever the behavior is.
BEHAVIOR = "sycophancy"
BEHAVIORS = ["sycophancy", "myopic-reward"]

# Prompt: optional text appended to every question (decided with src/format_check.py)
ANSWER_INSTRUCTION = "\n\nAnswer with (A) or (B)."
INSTRUCTION = ANSWER_INSTRUCTION  # chosen Oct 26 from format_check on 1.5B

# Experiment settings
SEED = 0
CONSTRUCT_FRAC = 0.4  # 40% construction (400 of 1000 for sycophancy), the rest evaluation
STEER_FROM = "all"    # every position; "reply" (CAA-style) was too weak on Qwen-1.5B (Δm <= 0.22)
# Layer and alpha are chosen per (model, behavior) by src/sweep.py and stored in <run_dir>/setup.json.
# Vectors are always letter-balanced (½ mean of A-pairs + ½ mean of B-pairs), and pairs are sampled
# stratified by letter, because each diff has a large "(A vs (B" token part that must cancel.


def data_path(behavior=BEHAVIOR):
    return DATA_DIR / f"{behavior}.json"


def split_path(behavior=BEHAVIOR):
    return DATA_DIR / f"split_{behavior}.json"


def run_dir(model=MODEL, behavior=BEHAVIOR):
    """Folder for one (model, behavior) run, e.g. results/Qwen2.5-1.5B-Instruct_sycophancy/."""
    d = RESULTS / f"{model.split('/')[-1]}_{behavior}"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def load_setup(model=MODEL, behavior=BEHAVIOR):
    """{"layer", "alpha", "steer_from", ...} chosen by the sweep for this run."""
    path = run_dir(model, behavior) / "setup.json"
    assert path.exists(), f"no {path}; run `python -m src.sweep --model {model} --behavior {behavior}` first"
    return json.loads(path.read_text())


def save_setup(setup, model=MODEL, behavior=BEHAVIOR):
    (run_dir(model, behavior) / "setup.json").write_text(json.dumps(setup, indent=1))
