# Lab Record: Assignment 3, ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Lab 3: Text Generation Using GPT-2

**Aim.** Use GPT-2 to generate continuations, compare decoding settings, and
fine-tune the last Transformer block to write short study tips.

**Software / model.** Hugging Face Transformers, `openai-community/gpt2`
(124M parameters). Script: `lab3_gpt2_text_generation.py` (outputs are also
saved to `lab_outputs/lab3_decoding.csv`).

**Parameters.** `max_new_tokens = 45`. Sampling uses top_p = 0.9, T ∈ {0.6,
1.0} and seeds {42, 43}. Fine-tuning: only `transformer.h[-1]` is trainable
(about 7.1M parameters), AdamW, lr = 2e-5, 3 epochs over 6 examples,
prompt tokens masked with −100.

### Decoding observation table (0 = poor, 1 = partial, 2 = strong)

| Prompt | Decoding and seed | Coherence 0–2 | Relevance 0–2 | Creativity 0–2 | Supporting excerpt |
|--------|-------------------|---------------|---------------|----------------|--------------------|
| Robot story | Greedy | | | | |
| Robot story | T 0.6, seed 42 | | | | |
| Robot story | T 0.6, seed 43 | | | | |
| Robot story | T 1.0, seed 42 | | | | |
| Robot story | T 1.0, seed 43 | | | | |
| Library story | Greedy | | | | |
| Library story | T 0.6, seed 42 | | | | |
| Library story | T 0.6, seed 43 | | | | |
| Library story | T 1.0, seed 42 | | | | |
| Library story | T 1.0, seed 43 | | | | |

**Typical pattern.** Greedy output is fluent but tends to repeat a phrase or
loop (high coherence, low creativity). T = 0.6 stays on topic with some
variety. T = 1.0 is the most novel but drifts off topic or loses grammar
more often. The more specific library prompt keeps all methods more
relevant than the open-ended robot prompt.

### Held-out comparison (relevance 0–2 + usefulness 0–2)

| Held-out topic | Base score / 4 | Tuned score / 4 | Example difference |
|----------------|----------------|-----------------|--------------------|
| Recursion | | | |
| Matrices | | | |
| Binary search | | | |

**Expected behaviour.** The last-example loss falls only a little over 3
epochs (for example from about 4–5 to about 3–4). With six examples and a
small learning rate, the tuned model may pick up the *format* (one short
imperative sentence ending at EOS) more than the *content*. Base GPT-2
often rambles, or continues the "Topic:" pattern with new topics. Any
improvement is small and inconsistent. As the manual notes, a six-example
fine-tune shows the workflow; it is not evidence of reliable improvement.

### Result and limitation

Decoding strategy controls a trade-off between safe and repetitive output
(greedy, low T) and varied but risky output (high T). **Limitation:** the
dataset is tiny, so the tuned outputs can be generic or factually empty.
**Next step:** use 100+ varied, approved pairs, a validation split, and more
epochs.

### Viva answers

1. **How does greedy decoding select the next token?**
   At each step it takes the arg-max of the model's next-token
   distribution, appends that token, and repeats. It is deterministic, so
   seeds have no effect. It often falls into repetitive loops, because the
   locally most likely token is not always part of the best sequence.

2. **How does temperature affect sampling?**
   The logits are divided by T before the softmax. T < 1 sharpens the
   distribution towards the most likely tokens (more conservative). T > 1
   flattens it so less likely tokens are picked more often (more diverse,
   less coherent). Here top-p = 0.9 is also applied, which samples only
   from the smallest set of tokens whose total probability reaches 0.9.

3. **Which tokens contribute to the fine-tuning loss?**
   Only the answer tokens: the study tip and the final EOS. Prompt tokens
   ("Topic: …\nStudy tip:") get the label −100, which the cross-entropy
   loss ignores. So the model learns to *answer* the prompt, not to
   reproduce it, and it learns to stop at EOS.

4. **Why are the held-out topics excluded from training?**
   To measure generalisation. If recursion, matrices or binary search were
   in the training set, a good score could just mean the model memorised
   the answer. Unseen topics test whether the model learned the task.
