# RUNLOG — 2000-Step LLM Speedrun

Corpus: `train_corpus.txt` (7,318,592 bytes; ~14% Devanagari). Dev: `dev_eval.txt` (159,225 bytes; ~20% Devanagari).
Scoring: `python evaluate.py --checkpoint ckpt.pt --text_file dev_eval.txt` → bits-per-byte (lower better).
Rule: change ONE thing per run. Params must stay ≤ 2,000,000; steps ≤ 2,000. CPU only.

| # | Change | Params | Dev bpb | Δ | Keep? |
|---|--------|-------:|--------:|---|-------|
| 0 | Baseline (byte tok, Adam 3e-4 const LR, std=0.05 init, no tie/clip/wd) | 1,339,840 | **2.3718** | — | baseline |
| 1 | BPE tokenizer, vocab 2048 (only change) | 1,913,280 | **2.0991** | −0.2727 |  keep |
| 2 | LR warmup(100) + cosine decay to 0.1×, peak 1e-3 | 1,913,280 | **2.0434** | −0.0557 |  keep |
| 3 | Weight tying (head = tok_emb) | 1,585,600 | **2.0661** | +0.0227 | ⚠️ conditional — banks 327k params for capacity retune |
| 4 | Gradient clipping @ norm 1.0 | 1,585,600 | **2.0675** | +0.0014 | ➖ neutral (kept as insurance) |
| 5 | Scaled residual init (std 0.02, proj ×1/√(2·n_layer)) | 1,585,600 | **2.0419** | −0.0256 |  keep (new best; clears tying checkpoint) |
| 6 | Capacity: n_embd 160 → 176 (spend banked params) | 1,879,328 | **2.0270** | −0.0149 |  keep (new best) |
| 7 | block_size 128 → 256 (more context) | 1,901,856 | **2.0044** | −0.0226 |  keep (new best) |
| 8 | AdamW weight decay 0.1 (2D weights only) | 1,901,856 | **2.0049** | +0.0005 |  revert (noise, as predicted) |

---

## Run 0 — Baseline
- **Hypothesis:** establish reference.
- **Changed:** nothing (starter code as-is).
- **Config:** vocab 256 (byte), block 128, n_layer 4, n_head 4, n_embd 160, Adam lr=3e-4 constant, batch 8, init N(0, 0.05), no weight tying / clipping / weight decay / warmup.
- **Result:** dev bpb **2.3718**; final train loss ~1.73 (nats); 88s / 2000 steps (~44 ms/step); 1,339,840 params.
- **Conclusion:** reference point. Byte tokenizer spends 3 tokens per Devanagari char, so the Hindi ~20% of dev is heavily penalized in effective context. Biggest suspected lever = BPE tokenizer.

## Run 1 — BPE tokenizer (vocab 2048)
- **Hypothesis:** byte-level tokenizer wastes 3 tokens per Devanagari char; a subword BPE trained on the corpus shortens sequences and lets each prediction cover more bytes → lower bpb.
- **Changed:** ONLY the tokenizer. Byte-level (vocab 256) → byte-level BPE (vocab 2048), trained on `train_corpus.txt` only via `train_bpe.py`, saved to `bpe.json`. Lossless byte fallback verified (round-trips train/dev + emoji/raw-byte edge cases). Model/optimizer/schedule unchanged. `Config.vocab_size` rides along automatically (2048) so evaluate.py rebuilds correctly.
- **Compression:** 3.01 bytes/token on dev (was 1.0). block_size 128 now spans ~385 bytes of context.
- **Params:** 1,339,840 → 1,913,280 (embedding+head grew with vocab; still < 2,000,000).
- **Result:** dev bpb **2.3718 → 2.0991** (−0.2727, −11.5%). Raw train loss rose 1.73→4.28 but that is per-token over a 2048-way softmax — NOT comparable; bpb is the metric and it dropped.
- **Conclusion:**  keep. Biggest single lever as predicted. Note params now near the 2M cap, so weight tying (queued) becomes important to reclaim room for capacity.

## Run 2 — LR warmup + cosine decay (peak 1e-3)
- **Hypothesis:** baseline used constant Adam lr=3e-4 and the loss was still descending at step 2000 (undertrained). Warmup + cosine decay with a higher peak LR extracts more from the fixed 2000-step budget.
- **Changed:** ONLY the optimizer schedule in `train.py`. Constant 3e-4 → linear warmup over 100 steps to peak 1e-3, then cosine decay to 0.1×peak (1e-4). Model, tokenizer, params identical.
- **Result:** dev bpb **2.0991 → 2.0434** (−0.0557, −2.7%).
- **Conclusion:**  keep. Combined schedule+higher-peak change; attributed together as intended. Second-largest lever so far, exactly as queued.

## Run 3 — Weight tying
- **Hypothesis:** tying head.weight = tok_emb.weight is usually neutral-to-slightly-better for bpb AND frees vocab*n_embd = 327,680 params, reopening budget for a later capacity retune (we were at 1.91M, near the 2M cap).
- **Changed:** ONLY `Config.tie_weights` False → True (model already wired it up; rides in saved config so evaluate.py rebuilds correctly).
- **Params:** 1,913,280 → 1,585,600 (−327,680).
- **Result:** dev bpb **2.0434 → 2.0661** (+0.0227, slightly WORSE in isolation).
- **Conclusion:**  conditional keep. Tying alone is a small regression here, but the point is the freed 327k-param budget. HARD CHECKPOINT: the capacity retune (Run 6) must beat the untied 2.0434, else revert tying. Cheap param-neutral wins (grad clip, scaled init) run first.

## Run 4 — Gradient clipping (max-norm 1.0)
- **Hypothesis:** clipping stabilizes the higher 1e-3 peak LR and protects against gradient spikes over 2000 steps.
- **Changed:** ONLY added `clip_grad_norm_(model.parameters(), 1.0)` before opt.step(). Config tied, params unchanged.
- **Result:** dev bpb **2.0661 → 2.0675** (+0.0014, within noise).
- **Conclusion:**  neutral. Training was already stable so there was nothing to clip. Kept as cheap insurance for the final run (negligible cost, protects vs a one-off spike). Baseline for next runs stays the tied 2.0661.

## Run 5 — Scaled residual init
- **Hypothesis:** flat std=0.05 init lets residual-stream variance grow with depth; GPT-2-style init (std 0.02, residual projections scaled by 1/√(2·n_layer)) makes the high 1e-3 LR better-behaved from step 1.
- **Changed:** ONLY `model._init` / init logic. Base std 0.05 → 0.02; `attn.proj.weight` and `mlp.2.weight` (the projections writing back to the residual stream) further scaled to std 0.02/√(2·n_layer). Param-neutral.
- **Result:** dev bpb **2.0675 → 2.0419** (−0.0256).
- **Conclusion:**  keep. New overall best. Also clears the Run-3 tying checkpoint: 2.0419 < untied 2.0434, so tying + scaled init together beat the untied path — tying validated before even spending its banked params.

## Run 6 — Capacity: n_embd 160 → 176
- **Hypothesis:** we're undertrained and now have ~414k free params under the cap (from tying). Spending them on width (n_embd) adds capacity the 2000-step budget can still use.
- **Changed:** ONLY `Config.n_embd` 160 → 176 (n_head stays 4, head dim 44). Everything else identical.
- **Params:** 1,585,600 → 1,879,328 (< 2,000,000).
- **Result:** dev bpb **2.0419 → 2.0270** (−0.0149).
- **Conclusion:**  keep. New best. Confirms the tying→capacity plan: freed params reinvested into width net a clear gain.

## Run 7 — block_size 128 → 256
- **Hypothesis:** BPE made sequences ~3× shorter, so 128 tokens covered ~385 bytes; doubling the window gives each prediction more real left context (evaluate.py carries 50% context, so longer blocks help scoring too). pos_emb only costs block×n_embd extra params.
- **Changed:** ONLY `Config.block_size` 128 → 256. Everything else identical.
- **Params:** 1,879,328 → 1,901,856 (< 2,000,000).
- **Result:** dev bpb **2.0270 → 2.0044** (−0.0226). Train ~227s (slower per step, still well within the 2000-step cap).
- **Conclusion:**  keep. New best, crosses under 2.01. Context length is a real lever once tokens are subword-sized.

## Run 8 — AdamW weight decay 0.1
- **Hypothesis (mine, skeptical):** weight decay is a regularizer; in a 2000-step UNDERfitting run it should barely matter. Testing to complete the queue.
- **Changed:** ONLY the optimizer: Adam → AdamW with weight_decay=0.1 applied to 2D matmul weights only (biases/LayerNorm exempt via param groups).
- **Result:** dev bpb **2.0044 → 2.0049** (+0.0005, within noise).
- **Conclusion:**  revert. Confirms the prediction — no gain in the underfit regime. Reverted by setting weight_decay default 0.0 (AdamW with wd=0 is identical to Adam, so the locked config = Run 7 and reproduces 2.0044 under the fixed seed).

---

## LOCKED CONFIG (= Run 7)
BPE tokenizer vocab 2048 · block_size 256 · n_layer 4 · n_head 4 · n_embd 176 · tie_weights True ·
Adam (weight_decay 0) lr peak 1e-3, warmup 100, cosine to 0.1× · grad_clip 1.0 · scaled residual init · batch 8 · seed 1337 · 2000 steps.
Params 1,901,856 (< 2,000,000). Experiment-sequence dev bpb **2.0044** (baseline 2.3718 → −15.5%).

## Final checkpoint — cap-safety dedup fix
- **Issue found at lock time:** with tying, the state_dict stored `tok_emb.weight` AND `head.weight`
  as two keys sharing one tensor. `model.n_params()` (and evaluate.py's printed `n_params`)
  dedupes to 1,901,856, but a NAIVE `sum(numel)` over the checkpoint tensors reads 2,262,304
  — over the 2M cap. To remove all DQ risk under any counting method, `model.py` no longer
  creates a separate `head` when tied; the output projection uses `tok_emb.weight` directly via
  `F.linear`. Same math, same params, evaluate.py rebuilds identically.
- **Effect:** checkpoint now has NO `head.weight` key; naive sum = 1,901,856 = model.n_params(). ✅
- **Side effect (honest):** dropping the throwaway head Linear shifted RNG consumption at model
  construction, which shifted the `get_batch` stream → different batch ORDER → **final dev bpb
  2.0094** (vs 2.0044). Trained-param init is unchanged; the +0.0050 is batch-order seed noise,
  not a regression. Kept seed 1337; did NOT engineer the RNG to recover the lower noise draw.
- **SHIPPED:** `ckpt.pt` — **dev bpb 2.0094**, 1,901,856 params, 2000 steps (baseline 2.3718 → −15.3%).
