# CSU coding plan

Oct 22, 2026 · @Phu

Eight milestones, each small enough to finish and check in one sitting: 1 to 4 on Thursday, 5 to 8 on Friday. Read the [CSU project primer](https://claude.ai/artifact/4yhdLW1TLSp5gvHxXMboMY) first; this plan assumes its terms.

## 0. Setup (about 30 min)

Write code as plain .py files in VS Code, debug it on your laptop with a tiny model, and run the real thing on a Colab GPU.

**Two ways to run on Colab from VS Code (pick one):**

- Install Google's **Colab extension for VS Code**, open your notebook, and choose a Colab runtime as the kernel. Your laptop's editor uses Colab's GPU.
- Or push the repo to GitHub, and in a Colab notebook run `!git clone ...` then `%cd` into it and import from `src/`.

**Debug small first.** Qwen2.5-0.5B-Instruct has the same architecture as the 1.5B and runs on a laptop CPU. Make every script take `--model` and `--limit` flags, get it working on 0.5B with `--limit 20`, then switch to 1.5B on Colab. This avoids burning compute units on bugs.

**Repo layout:**

```
csu-steering/
  CLAUDE.md            # project summary for Claude Code (paste primer sections 3-4)
  requirements.txt     # torch transformers numpy scikit-learn matplotlib pandas tqdm
  data/                # sycophancy generate_dataset.json from the CAA repo
  src/
    config.py          # MODEL, LAYER, ALPHA, SEED, paths
    data.py            # milestone 1
    model.py           # milestones 2-3: load, capture activations, steering hook
    sweep.py           # milestone 4
    labels.py          # milestone 5
    csu.py             # milestone 6 (numpy only, no GPU)
    analysis.py        # milestones 7-8
  results/             # cached .pt / .npz files and figures
  run_colab.ipynb      # thin notebook that calls src/ on the GPU
```

Save every expensive result to `results/` (activations, Δm) so analysis never needs the GPU again. Colab disconnects, so save to Google Drive or download after each run.

## Thursday: get steering working

By Thursday night you should have a steering vector that moves the model the right way on average, and a chosen layer and α.

### Milestone 1: data (`data.py`)

- Load the 1,000 questions, shuffle with a fixed seed, and split into 400 construction / 600 evaluation.
- `build_prompt(question)`: use `tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)` and append `"("`, so the model's next token is the answer letter.
- For each item, store `syc_letter` (from `answer_matching_behavior`, e.g. "A") and `other_letter`.
- **Done when:** the counts are 400 / 600, about half of the sycophantic answers are A (print the share), and one printed prompt looks right.

### Milestone 2: record hidden states (`model.py`)

- Load the model in float16 on the GPU, call `model.eval()`, and wrap everything in `torch.no_grad()`.
- `get_hidden(text)`: run with `output_hidden_states=True`. `hidden_states[l + 1]` is the output of layer l (index 0 is the embeddings). Keep only the **last token**.
- **Construction:** for each of the 400 pairs, run `prompt + syc_letter` and `prompt + other_letter`, and save `diffs[i, layer, :] = h_syc − h_other` for all layers. That's about 400 × 28 × 1536 floats, around 70 MB.
- **Evaluation:** for each of the 600 prompts, save h(x) at the last token for all layers, plus `m_base = logit[syc_letter] − logit[other_letter]`.
- **Done when:** the saved shapes are right, and `tokenizer.tokenize(prompt + "A")[-2:]` shows `"("` and `"A"` as separate tokens.

### Milestone 3: steering hook (`model.py`)

- `add_steering(layer, vector, alpha)`: register a forward hook on `model.model.layers[layer]` that adds `alpha * vector` to the layer output at every position. The output may be a tuple (edit `output[0]`) or a tensor, so handle both. Return the handle so you can call `.remove()`.
- `delta_m(prompt, vector, alpha)`: m with steering minus m without.
- **Done when:** α = 0 gives Δm = 0 (within 1e-3), and after removing the hook the model reproduces `m_base` exactly.

### Milestone 4: choose layer and α (`sweep.py`)

- Build v̄ = mean of all 400 diffs at each layer, and use 100 evaluation prompts.
- **Layer sweep:** layers 6, 8, …, 18 at α = 1. Plot mean Δm and the failure rate (share with Δm < 0) per layer, and keep the layer with the largest mean Δm.
- **α sweep:** at that layer, try α in {0.5, 1, 2, 4}. Also track the probability of A + B together: if it drops below about 0.5, α is breaking the model.
- Pick α where mean Δm is clearly positive and the failure rate is between 5% and 40%, and write LAYER and ALPHA into `config.py`.
- **Done when:** you have both plots saved. These go in your write-up as setup figures.

## Friday: test the hypothesis

By Friday night you should have the two figures for your slides: an AUROC table and a risk–coverage plot. Only milestone 5 needs the GPU.

### Milestone 5: failure labels (`labels.py`, GPU, about 10 min)

- Settings: n in {25, 50, 100} construction pairs × 3 random seeds, plus n = 400 once. That's 10 settings.
- For each setting, pick n random construction pairs, build v̄ at LAYER, steer all 600 evaluation prompts, and record Δm and `failure = Δm < 0`. Also save which construction indices were used, because milestone 6 needs them.
- Save everything to `results/labels.npz`.
- **Done when:** you've printed the failure rate per n. Smaller n should give more failures. **Fallback:** if failures are under 3% everywhere, redefine failure as "Δm in the bottom 20%" (weak steering) and say so in the write-up.

### Milestone 6: CSU and baselines (`csu.py`, numpy only)

- For each setting, resample its n indices with replacement B = 50 times and average the matching diffs to get V (50 × 1536).
- For the 600 evaluation hidden states H (600 × 1536): `P = H @ V.T`, then `csu = P.var(axis=1, ddof=1)`.
- Also compute `proj = H @ v_bar`, `rel_csu = sqrt(csu) / (abs(proj) + 1e-6)`, `norm = ‖h‖`, and `conf = abs(m_base)`.
- Global context per setting: the mean cosine similarity between the bootstrap vectors and v̄.
- **H1 check:** recompute CSU with B = 20 and B = 100. If the Spearman correlation with B = 50 is above 0.9, B = 50 is stable enough.
- **Done when:** `results/features.csv` has one row per (setting, prompt), with columns n, seed, prompt\_id, failure, delta\_m, proj, norm, conf, csu, rel\_csu.

### Milestone 7: does CSU predict failure? (`analysis.py`)

- First, the simple check: AUROC of each signal on its own, per n (`roc_auc_score(failure, feature)`; if it's below 0.5, the signal works in reverse, so report 1 − AUROC and note the direction).
- Then fit logistic regression, standardizing features and using log(csu) because it is skewed. Use 5-fold `GroupKFold` with groups = prompt\_id, so the same prompt never appears in both training and test folds.
- Compare 4 models: projection only / baselines (proj + norm + conf) / CSU only / baselines + CSU + rel\_csu.
- **Done when:** you have a table with one row per n and the AUROC and AP of each model. The headline is whether "baselines + CSU" beats "baselines".

### Milestone 8: selective steering (`analysis.py`)

- Use the cross-validated risk predictions from milestone 7. For coverage c = 10%, 20%, …, 100%, steer only the c share of prompts with the lowest predicted risk, and measure the failure rate among them.
- Plot the failure rate against coverage with one line each for random, baselines, CSU only, and baselines + CSU. Use n = 50, or whichever n shows the clearest picture.
- Summarize each line with one number: the area under the risk–coverage curve (lower is better).
- **Done when:** the figure is saved. This is slide 2.

## Pitfalls

Most bugs in this kind of project are silent: the code runs, but the numbers mean something else. These checks catch the common ones.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Mean Δm is negative at every layer | Vector sign flipped | v = h\_syc − h\_other, and +α should raise m |
| Steering does nothing at any layer | Off-by-one in `hidden_states`, or the hook is on the wrong module | Hook `model.model.layers[L]` and read `hidden_states[L + 1]` |
| The vector just means "say A" | Sycophantic answer is A in most items | Check the A/B share in milestone 1 and balance it if it's skewed |
| NaN or garbage logits | float16 overflow or a dtype mismatch in the hook | `vector.to(output.dtype).to(output.device)`, and try a smaller α |
| AUROC looks too good (above 0.95) | Leakage | Construction and evaluation sets never overlap, and GroupKFold uses prompt\_id |
| Lost a whole run | Colab disconnected | Save to `results/` after every milestone and copy to Drive |

One thing to mention in the write-up: construction activations are read at the answer-letter token, while evaluation activations are read at the "(" just before it. This is standard CAA practice, but a reviewer may ask about it.

## Using Claude Code well

Give Claude Code one milestone at a time, and don't move on until you can explain what the code does.

1. Put the primer's sections 3 and 4, plus this plan, into `CLAUDE.md` so every session knows the project.
2. Paste one milestone's bullets as the task, for example: *"Implement Milestone 2 in src/model.py. Keep it simple and commented. Test with Qwen2.5-0.5B-Instruct and --limit 5."*
3. Ask it to run the milestone's **Done when** check and show you the output.
4. Ask *"explain this file line by line as if I'm new to PyTorch"* until you could rewrite the key lines yourself. This is your Q&A preparation.
5. Commit after each milestone (`git commit -m "milestone 2: activations"`) so you can always go back.

If you get stuck on a result (for example, no failures, or CSU looks random), paste the numbers here and I'll help you work out what they mean.
