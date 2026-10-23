# Progress

Deadline: **Tue Oct 27, 11:59 PM PT** (one upload on E-learning).

## Milestones

| # | Milestone | File | Status | Done-when result |
|---|-----------|------|--------|------------------|
| 0 | Setup: repo, env, data, Colab | — | 🟡 in progress | |
| 1 | Data split + prompts | `src/data.py` | ⬜ | |
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
- [ ] Python env + `pip install -r requirements.txt`
- [ ] Download `generate_dataset.json` into `data/`
- [ ] Colab runtime working (VS Code extension or git clone)
- [ ] Push to GitHub (needed for Colab clone)

## Log

Notes, decisions, and odd results. Newest first.

- **Oct 23** — project set up.
