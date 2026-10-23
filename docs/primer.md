# CSU project primer

Oct 22, 2026 · @Phu

## 1. The big picture

Activation steering changes how a model behaves by adding a fixed vector to its internal numbers while it runs, with no retraining.

**Hidden states.** As a prompt passes through the model, each layer turns every token into a long list of numbers (1,536 numbers per token for Qwen2.5-1.5B). This list is the *hidden state* or *activation*, written h(x). It is the model's internal summary of the text so far.

**A behavior is roughly a direction.** If you take prompts where the model behaves one way and prompts where it behaves the other way, their hidden states tend to differ along a consistent direction. Pushing a hidden state along that direction pushes the behavior.

**CAA (Contrastive Activation Addition)** is the simplest way to find that direction:

1. Take n pairs of examples. Each pair is the same A/B question, once ending in the sycophantic answer and once in the honest answer.
2. Run both through the model and record the hidden state at one layer, at the answer token.
3. Subtract: honest minus sycophantic (or the other way round). Average the n differences. That average is the steering vector v.
4. At test time, add α × v to the hidden state at that layer. A positive α pushes towards the behavior, a negative α pushes away.

This is the method in [Rimsky et al., 2024](https://aclanthology.org/2024.acl-long.828/), your reference \[3\].

## 2. The problem

Steering works on average but backfires on some individual prompts, and nobody can tell in advance which ones.

Tan et al. \[4\] and Braun et al. \[5\] found that the same vector can push one prompt strongly the right way, barely move another, and push a third the *wrong* way. Braun et al. relate this to properties of the whole training set for the vector, such as how much the n difference vectors agree with each other. But those are **one number per vector**: they say whether a vector is reliable overall, not which test prompt it will fail on.

Your project asks the per-prompt question: *for this particular input, should I trust the steering or skip it?*

## 3. Your idea: construction-sample uncertainty (CSU)

If the steering vector would come out noticeably different with a slightly different set of training pairs, its effect on a given prompt is shaky, and shaky prompts may be the ones that backfire.

**Bootstrap, in plain words.** You have n training pairs. Draw n pairs *with replacement* (some appear twice, some not at all) and build a vector from them. Repeat B times (say B = 50). You now have 50 slightly different steering vectors v₁ … v₅₀, which show how much the vector depends on the luck of which examples you picked.

**Making it per-prompt.** For a test prompt x, compute how strongly each of the 50 vectors lines up with its hidden state (a dot product, h(x)ᵀv\_b). If those 50 numbers are close together, every version of the vector agrees about this prompt. If they are spread out, the vectors disagree about it. CSU is that spread:

```latex
U_{\mathrm{CSU}}(x) = \mathrm{Var}_{b}\left[\, h(x)^{\top} v_b \,\right]
```

**Why it's cheap.** The 50 vectors are just averages of difference vectors you already stored. You run the model once per example to record hidden states; the bootstrap itself is a few lines of numpy and takes seconds.

**One trap to check.** Prompts with a larger hidden state (bigger ‖h(x)‖) get a larger spread automatically. So you also test a *relative* CSU (spread divided by the size of the average effect) and ‖h(x)‖ on its own as a baseline, so you can show CSU is not just measuring size.

## 4. The experiment, concretely

One model (Qwen2.5-1.5B-Instruct), one behavior (sycophancy), one question: does CSU flag the prompts where steering backfires better than simpler signals do?

**Data.** The CAA sycophancy set: 1,000 A/B questions where a user describes themselves and one answer agrees with their likely view ([generate\_dataset.json](https://github.com/nrimsky/CAA/tree/main/datasets/generate/sycophancy)). The official test file has only 50 questions, too few for AUROC, so we split the 1,000 ourselves:

- 400 *construction* questions, used only to build steering vectors
- 600 *evaluation* questions, used only to measure failures (never both, so there is no leakage)

**Gold labels (computed, no annotation).** End each evaluation prompt just before the answer letter and read the model's scores (logits) for "A" and "B".

1. m(x) = score of the sycophantic letter minus score of the other letter.
2. Run once without steering (m\_base) and once with steering towards sycophancy (m\_steered).
3. Δm = m\_steered − m\_base. If Δm < 0, steering pushed the wrong way: **label 1 (failure)**. Otherwise label 0.

**Setup choices made on the data, not guessed.** Try layers 8 to 16 (of 28) and keep the one where steering moves m the most on average. Then pick α so steering clearly works on average but not on every prompt. Build vectors from n = 25, 50, 100 and 400 construction pairs: small n gives noisier vectors and more failures, which is where CSU should matter most.

**Per-prompt signals to compare.**

| Signal | What it is | Role |
| --- | --- | --- |
| Projection | h(x)ᵀv, how much the prompt already lines up with the vector | Baseline |
| Activation norm | ‖h(x)‖ | Baseline (the trap in section 3) |
| Base confidence | \|m\_base\|, how sure the model already is | Baseline |
| CSU | Spread of h(x)ᵀv\_b over the 50 bootstrap vectors | Your method |
| Relative CSU | CSU spread divided by \|projection\| | Your method |

Global reliability measures from \[5\] (agreement among difference vectors) are one number per vector, so they cannot rank prompts. We report them per n as context.

**Metrics.**

- **AUROC / average precision:** fit a small logistic regression with 5-fold cross-validation, once on baselines only and once on baselines + CSU. The key result is whether adding CSU raises AUROC.
- **Risk–coverage curve:** sort prompts from most to least trusted, steer only the top fraction, and plot the failure rate against the fraction steered. Compare CSU ordering with projection, confidence and random ordering. This is the plot for your slide.

If CSU adds nothing, that is a legitimate finding, and the proposal already says so.

## 5. Answering the reviewers

Both gaps the reviewers raised are closed by section 4; say so explicitly in the write-up.

| Reviewer comment | Answer |
| --- | --- |
| "The dataset is not described." | CAA sycophancy A/B set, 1,000 questions, split 400 construction / 600 evaluation with no overlap. |
| "Unclear how gold labels will be obtained." | Computed from the model's own logits: a prompt is a failure when steering lowers its preference for the target answer (Δm < 0). No human labeling needed. |

What changed from the proposal: one model instead of two, and robustness and open-ended safety tests are moved to future work. The course says it's fine for a project to evolve.

## 6. Schedule

All experiments run Thursday and Friday so the busy weekend is only a buffer, and the results are locked by Sunday night.

1. **Thu Oct 22:** read this primer, then notebook part 1: load the model, record hidden states, pick the layer and α, compute Δm labels. Check: the failure rate is somewhere between 5% and 40%.
2. **Fri Oct 23:** notebook part 2: bootstrap, CSU, baselines, AUROC table, risk–coverage plot.
3. **Sat–Sun Oct 24–25 (light):** look at the results, rerun anything odd, save the final figures.
4. **Mon Oct 26:** 2 slides plus a 2-minute script, then the write-up draft on the course template.
5. **Tue Oct 27:** final pass, then submit on E-learning before 11:59 PM PT (one upload only).

Compute: Colab Pro is plenty. The 1.5B model needs about 4 GB of GPU memory, and the whole run should take well under an hour on an L4 or A100.

## 7. Questions you should be able to answer

If you can answer these in one or two sentences each, you're ready for the Q&A.

- [ ] Why add a vector instead of fine-tuning? *(Cheap, reversible, no training, and it's applied only at inference time.)*
- [ ] What exactly is your failure label? *(Δm < 0: steering lowered the model's preference for the target answer.)*
- [ ] Why bootstrap? *(It shows how much the vector depends on which training pairs happened to be chosen, with no extra data.)*
- [ ] Isn't CSU just measuring ‖h(x)‖? *(We tested that: a norm baseline and a relative CSU.)*
- [ ] Why not use Braun et al.'s reliability measures? *(They give one number per vector, so they can't say which prompt will fail.)*
- [ ] What would you do with more time? *(More behaviors, larger models, paraphrase robustness, open-ended generation.)*

**Glossary.** *Logit:* the model's raw score for a token, before it becomes a probability. *Residual stream:* the running hidden state that each layer reads from and adds to. *AUROC:* the chance that a random failure gets a higher risk score than a random success (0.5 = guessing, 1.0 = perfect). *Coverage:* the fraction of prompts you choose to steer.
