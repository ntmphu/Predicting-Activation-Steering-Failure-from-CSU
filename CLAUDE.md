# CSU steering project

Does construction-sample uncertainty (CSU) predict, per prompt, when activation steering backfires?
Model: Qwen2.5-1.5B-Instruct (debug with Qwen2.5-0.5B-Instruct). Behavior: sycophancy (CAA A/B set).

Full context: [docs/primer.md](docs/primer.md) (concepts) and [docs/coding-plan.md](docs/coding-plan.md) (milestones).
Current status: [PROGRESS.md](PROGRESS.md) — read it first, update it after each milestone.

## Key definitions

- Steering vector v̄ = mean of (h_syc − h_other) over n construction pairs, at layer LAYER.
- m(x) = logit[syc_letter] − logit[other_letter] at the token after "(".
- Δm = m_steered − m_base. **Failure label = Δm < 0.**
- CSU(x) = Var_b[h(x)ᵀ v_b] over B = 50 bootstrap vectors. rel_csu = sqrt(csu) / (|proj| + 1e-6).
- Baselines: proj = h(x)ᵀv̄, norm = ‖h(x)‖, conf = |m_base|.
- Data split: 400 construction / 600 evaluation, fixed seed, never overlapping.
- `hidden_states[L + 1]` = output of layer L; hook `model.model.layers[L]`.

## Layout

```
src/        config.py data.py model.py sweep.py labels.py csu.py analysis.py
data/       generate_dataset.json (CAA sycophancy)
results/    cached .pt/.npz/.csv (gitignored), figures/ (committed)
docs/       primer, coding plan, proposal
run_colab.ipynb   thin notebook calling src/ on GPU
```

## Working rules

- One milestone at a time. Run its **Done when** check and show the output before moving on.
- Every script takes `--model` and `--limit`. Test on 0.5B with `--limit 20` before using the GPU.
- Save every expensive result to `results/` so analysis never needs the GPU again.
- Keep code simple and commented; the user must be able to explain every line in Q&A.

## Commit rules

- Short, concise, lowercase, imperative: `milestone 2: record activations`, `fix hook dtype`.
- One line, under ~50 chars. No body unless truly needed.
- **Never mention Claude/AI in commits. No `Co-Authored-By` trailer.** (Enforced by `.githooks/commit-msg`.)
- Commit after each milestone and after each meaningful fix. Only commit when the user asks.
