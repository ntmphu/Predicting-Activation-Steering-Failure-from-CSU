"""One-time move of the first run (Qwen2.5-1.5B, sycophancy) from results/ into its run folder.

Before the multi-run refactor, everything was saved directly in results/. Now each (model, behavior) run
has its own folder. Safe to run more than once. Run:  python scripts/migrate_results.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS, run_dir

OLD_TO_NEW = {
    "acts_Qwen2.5-1.5B-Instruct.pt": "acts.pt",
    "labels.npz": "labels.npz", "features.csv": "features.csv", "sweep.csv": "sweep.csv",
    "auroc_table.csv": "auroc_table.csv", "risk_coverage.csv": "risk_coverage.csv",
    "figures/sweep_all.png": "figures/sweep_all.png", "figures/sweep_reply.png": "figures/sweep_reply.png",
    "figures/risk_coverage_n25.png": "figures/risk_coverage_n25.png",
}
dest = run_dir("Qwen/Qwen2.5-1.5B-Instruct", "sycophancy")
for old, new in OLD_TO_NEW.items():
    src, dst = RESULTS / old, dest / new
    if src.exists():
        if dst.exists():
            src.unlink()  # already migrated (e.g. pulled again from the hub): drop the stale copy
        else:
            shutil.move(str(src), str(dst))
            print(f"moved {old} -> {dst.relative_to(RESULTS)}")

setup = dest / "setup.json"
if not setup.exists():  # the setup chosen on Oct 26 from src.sweep + src.diagnose
    setup.write_text(json.dumps({"layer": 18, "alpha": 3.48, "r": 0.1, "steer_from": "all",
                                 "chosen_by": "Oct 26 sweep + diagnose (balanced v̄)"}, indent=1))
    print(f"wrote {setup.relative_to(RESULTS)}")
for f in RESULTS.glob("acts_Qwen2.5-0.5B-Instruct_limit*.pt"):  # old debug files, cheap to recreate
    f.unlink()
print("migration done:", sorted(p.name for p in dest.iterdir()))
