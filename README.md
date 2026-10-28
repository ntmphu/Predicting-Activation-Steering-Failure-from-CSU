# Predicting Activation-Steering Failure from Construction-Sample Uncertainty

*"Can We Predict When Activation Steering Backfires?"*.

A steering vector is the mean difference of hidden states over n contrastive example pairs (CAA). Adding it to a
model's activations shifts behaviour on average, but pushes some inputs the wrong way. This project tests whether
**construction-sample uncertainty (CSU)**, the bootstrap variance of a vector's projected effect on one input,
predicts those per-input failures, and compares it with a **gradient-based** version (one backward pass per input).

- Models: Qwen2.5-1.5B-Instruct (main), Qwen2.5-3B-Instruct; Qwen2.5-0.5B-Instruct for debugging.
- Behaviours: sycophancy and myopic reward, A/B multiple-choice sets released with CAA (Rimsky et al., 2024).

## Key definitions

| Quantity | Definition |
|---|---|
| answer score | m(x) = logit("(" + target letter) − logit("(" + other letter) at the prompt's last position |
| steering effect | Δm = m_steered − m_base; **failure = Δm < 0** |
| steering vector | letter-balanced mean of h_target − h_other at layer L: ½ mean(A-target pairs) + ½ mean(B-target pairs) |
| CSU | Var_b[h(x)ᵀ v_b] over B = 50 stratified bootstrap vectors |
| gradient features | g(x) = Σ_t ∂m/∂h_{L,t}; prediction gᵀv̄, gCSU = Var_b[gᵀv_b], z = gᵀv̄ / sd_b(gᵀv_b) |

Qwen tokenizes "(A" and "(B" as single tokens (ids 4346 / 5349), so prompts end at the start of the assistant
reply and m reads those two tokens directly.

## Layout

```
src/
  config.py         paths, models, split sizes, prompt instruction
  data.py           load items, fixed 40/60 construction/evaluation split, prompt builder
  format_check.py   does the model answer with an option? (chooses the prompt instruction)
  model.py          record hidden states: construction diffs (answer token) + evaluation states (last token)
  steer.py          steering hook (add α·v at layer L) and its sanity checks
  sweep.py          layer × strength sweep; writes the chosen setting to setup.json
  diagnose.py       letter-confound diagnostic (plain vs letter-balanced vector)
  labels.py         steer every evaluation prompt for each construction size n and draw → Δm, failure
  csu.py            bootstrap vectors, CSU, baselines (+ gradient features if grads.pt exists)
  grad.py           g(x) by one backward pass per prompt
  analysis.py       cross-validated AUROC / AP, ΔAUROC bootstrap CIs, risk–coverage curves
  variability.py    many draws per n: which prompts change outcome across example sets, and what predicts it
  hub.py            back up / restore results/ to a private Hugging Face dataset repo
scripts/            prompt inspector, one-off results migration
data/               <behavior>.json (CAA A/B items) and split_<behavior>.json (fixed split, seed 0)
run_colab.ipynb     thin GPU runner that calls the scripts above in order
```

Every script takes `--model` and `--behavior`; most take `--limit` for quick tests. Results are written to
`results/<model>_<behavior>/` (activations, labels, features, AUROC tables, figures).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
.venv/Scripts/python -m pip install -r requirements.txt scipy huggingface_hub
```

Recording activations, steering and gradients need a GPU (one Colab L4/A100 is enough). Everything after that
(CSU, analysis, figures) runs on a CPU from the cached results.

## Reproducing the results

Open `run_colab.ipynb` on Colab with a GPU runtime and run the cells in order. The equivalent commands for one
(model, behavior) run are:

```bash
M=Qwen/Qwen2.5-1.5B-Instruct; B=sycophancy
python -m src.data          --model $M --behavior $B              # split + prompt checks
python -m src.model         --model $M --behavior $B              # hidden states            (GPU)
python -m src.sweep         --model $M --behavior $B --limit 100  # choose layer and α       (GPU)
python -m src.labels        --model $M --behavior $B              # failure labels, n = 25/50/100/all (GPU)
python -m src.grad          --model $M --behavior $B              # gradients                (GPU)
python -m src.csu           --model $M --behavior $B              # CSU, baselines, gradient features
python -m src.analysis      --model $M --behavior $B --grad       # AUROC tables, risk–coverage
python -m src.analysis      --model $M --behavior $B --grad --label construction
```

Repeated draws (8 example sets for each n = 10/16/25/50) and the instability analysis:

```bash
python -m src.labels      --model $M --behavior $B --ns 10,16,25,50 --seeds 8 --tag e2
python -m src.csu         --model $M --behavior $B --tag e2
python -m src.analysis    --model $M --behavior $B --tag e2 --grad
python -m src.variability --model $M --behavior $B --tag e2
```

Runs in the report: 1.5B sycophancy (all steps, including the repeated draws), 3B sycophancy and 1.5B myopic
reward (all steps except the gradients and repeated draws).

To test the pipeline quickly on a CPU, use `--model Qwen/Qwen2.5-0.5B-Instruct --limit 20`.

`python -m src.hub push|pull` backs up and restores `results/` using a Hugging Face token read from `HF_TOKEN`
(or prompted); the repo name is set in `src/config.py`. It is optional: every result can be regenerated with the
commands above.

## Data

The two datasets are the sycophancy and myopic-reward A/B sets distributed with
[Contrastive Activation Addition](https://github.com/nrimsky/CAA) (Rimsky et al., 2024), originally from
Anthropic's model-written evaluations (Perez et al., 2022). Each is split once (seed 0) into a construction set
(400 / 380 items) used only to build vectors and a disjoint evaluation set (600 / 570 items).
