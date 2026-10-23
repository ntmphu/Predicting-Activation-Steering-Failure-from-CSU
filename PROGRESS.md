# Progress

Deadline: **Tue Oct 27, 11:59 PM PT** (one upload on E-learning).

## Milestones

| # | Milestone | File | Status | Done-when result |
|---|-----------|------|--------|------------------|
| 0 | Setup: repo, env, data, Colab | — | ✅ | venv (CPU), `run_colab.ipynb` |
| 1 | Data split + prompts | `src/data.py` | ✅ | 400/600, syc=A 0.492/0.487, answer tokens clean 0/1200 bad |
| 2 | Record hidden states | `src/model.py` | ⬜ | |
| 3 | Steering hook | `src/model.py` | ⬜ | |
| 4 | Choose layer + α | `src/sweep.py` | ⬜ | LAYER = ?, ALPHA = ? |
| 5 | Failure labels (GPU) | `src/labels.py` | ⬜ | failure rate per n: |
| 6 | CSU + baselines | `src/csu.py` | ⬜ | H1 Spearman: |
| 7 | AUROC table | `src/analysis.py` | ⬜ | |
| 8 | Risk–coverage plot | `src/analysis.py` | ⬜ | |

Status: ⬜ todo · 🟡 in progress · ✅ done · ⚠️ blocked

## Schedule

- [ ] Thu Oct 22 — milestones 1–4
- [ ] Fri Oct 23 — milestones 5–8
- [ ] Sat–Sun Oct 24–25 — review results, rerun odd ones, final figures
- [ ] Mon Oct 26 — 2 slides + 2-min script, write-up draft
- [ ] Tue Oct 27 — final pass, submit

## Setup checklist

- [x] Git repo, CLAUDE.md, PROGRESS.md
- [x] Python env: `.venv` (torch 2.14.1+cpu, transformers 5.20.2), versions pinned
- [x] Download `generate_dataset.json` into `data/` (1,000 items, syc = A in 48.9%)
- [ ] Colab runtime working (VS Code extension or git clone)
- [x] Push to GitHub: https://github.com/ntmphu/Predicting-Activation-Steering-Failure-from-CSU

## Log

Notes, decisions, and odd results. Newest first.

- **Oct 23** — Milestone 1 done. **Deviation from coding plan:** Qwen tokenizes "(A" as ONE token,
  so prompts can't end with "(". Prompts end at `assistant\n`; m uses logits of tokens "(A" (4346) vs "(B" (5349).
  Verified on 0.5B and 1.5B: appending "(A"/"(B" never changes the prompt's tokens (0/2000 bad).
  Qwen's default system prompt is kept (mention in write-up). Split saved to `data/split.json` (seed 0).
- **Oct 23** — project set up.
