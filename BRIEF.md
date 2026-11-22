# Brief: proving that an AI model's answer is real, inside the ZisK zkVM

## 1. The problem
When you ask an AI service a question, you can't check which model answered, with which weights, or
whether the answer was altered. We want a **proof of inference**: a short certificate, checkable by
anyone in milliseconds, that says "this exact model, with these exact weights, given this input,
produced this output". No trust in whoever ran the model.

We build it on **ZisK**, a zkVM: you write the model as an ordinary Rust program, ZisK runs it, and it
produces a proof that the program ran correctly.

**The core difficulty is cost.** ZisK has no floating-point hardware. Every float multiply becomes
dozens of proven integer steps, and a language model is billions of float multiply-adds. Proving a
model naively in FP32 is far too expensive.

**A second requirement:** real services (e.g. OpenRouter) serve the **original, unquantized BF16
weights**. A proof about an 8-bit quantized copy doesn't prove what they serve. So we want to prove the
original model, at a cost that's practical.

Models used: **Qwen3-4B** (36 layers, 4B parameters, BF16) for the real benchmark;
**stories260K** (Karpathy's TinyStories, 260K parameters) for the demo and real GPU proofs.
Cost below = ZisK emulator cost (`ziskemu -X`), G = 10^9. Qwen numbers use the same 39-token prompt.

## 2. Progress so far
- Built and tested three ways of doing the model's math inside ZisK (Section 3).
- Measured the full Qwen3-4B model for two of them, and single layers for the third.
- Real GPU proofs for stories260K, both 8-bit and unquantized.
- A demo website ("provable story") where each turn of a shared story is proven.
- A memory harness so the benchmarks can't crash the test machine.
- All code cleaned and pushed (Section 5).

## 3. Approaches tried, and results

### A. Exact FP32 with a FloatDot precompile (baseline, original weights)
A precompile that computes FP32/BF16/FP16 dot products exactly as plain code would.
- Qwen3-4B full model, 39 tokens: **32.79 G steps, 809,902 G cost (about 810 T)**; outputs the reference
  next tokens [13, 3776, 1899, 11, 1340, 1730, 264, 25382].
- One layer: 22,004 G.
- Verdict: correct and unquantized, but far too expensive. A BF16 lookup-table variant was ruled out by
  the ZisK team (about 18 GB of memory per operation).

### B. 8-bit FP8 with the `dot_f8` precompile (main directive from the ZisK team)
Weights and activations are converted to 8-bit E4M3 with one scale per row. A custom state machine
proves 8-bit dot products as **exact integers** (8 multiply-adds per row, 1,024-row decode table, the
accumulator can't overflow), and the float scales are applied afterwards. Includes a "matrix mode" that does a
whole matrix in one call. Constraint tests accept 11 honest calls and reject 21 forgeries.
- Qwen3-4B full model, 39 tokens: **44.35 G steps, 7,677 G cost (about 7.7 T)**, same next tokens as FP32.
  That's **about 105× cheaper than exact FP32**.
- One layer: 198.8 G (the 7 weight matrix products are 115.7 G of that).
- Growth with tokens, one layer: 4 → 19.9 G, 39 → 198.8 G, 128 → 691.8 G, 256 → 1,489.4 G
  (about 5–6 G per token per layer, rising slowly as attention covers more tokens).
- stories260K real GPU proof ("Once upon a" + 2 tokens): 3.6 G, **427.5 s** on an RTX 4080 Laptop.
- Verdict: cheap and works end to end, but it proves an 8-bit copy, not the original weights.

### C. Unquantized via sumcheck (original BF16 weights, no new precompile)
A normal program outside ZisK does the matrix products on the original BF16 weights (summed exactly,
rounded once) and produces a **sumcheck proof** against a Ligero commitment to the weights. The ZisK
guest computes everything else in f32 itself (norms, RoPE, attention, softmax, SiLU) and **verifies
the sumcheck proofs against the activations it computed itself**. A wrong product makes the ZisK proof
impossible. The math runs outside; its correctness is checked inside.
- Correctness: exact-sum-then-round matches FP32's next tokens across all 36 layers; in every layer run
  so far the guest's output matched the prover's expected output.
- One layer, 39 tokens: **1,511 G** after fixes (was 1,659 G). That's about 7.6× the 8-bit layer, about 14.6× cheaper
  than the exact FP32 layer.
  - Where the cost goes: about 75% is verifying the proofs (FFT about 578 G, BN254 field arithmetic, column
    hashing about 161 G). Only about 233 G is the layer itself in f32.
  - Fixes applied: cheap activation encoding, not sending activations the guest already computes, and
    proving matrices in power-of-two row chunks so the prover fits in 12 GB (peak 8.2 GB). The chunking
    raised verification cost (13 proofs instead of 7), so the net gain was only 9%.
- Outside ZisK: about 30 s commit + 7.5 s prove per layer on 8 CPU threads.
- Growth with tokens, one layer: 4 → 1,043 G, 39 → 1,511 G, **128 → failed** (the guest errors out partway,
  cause not diagnosed yet; suspected the guest's 532 MB heap), 256 → pending.
  Early sign: sumcheck grows much more slowly per token than 8-bit, but starts from a much higher base.
- Full model: running. Projection about 36 × 1,511 G ≈ 54 T, **an estimate, not measured**.
- Verdict: the only practical path we have to prove the original weights, but currently about 7.6× the
  8-bit cost per layer, and it doesn't yet scale past 39 tokens.

### D. Unquantized stories260K directly in software FP32 (small model only)
For a 260K-parameter model, plain FP32 in the guest is affordable.
- "Once upon a" + 2 tokens: **16.18 G cost, 165.9 M steps, real GPU proof in 1,830 s** (verified, a tampered
  copy rejected). That's 4.5× the 8-bit cost and 4.3× its proof time.

### E. Demo: provable story website (8-bit stories260K)
The visitor adds a word, the model writes 2 sentences, and each turn is proven. The **KV cache is carried between proofs**
with hash links, so a turn only proves its new tokens: a late turn drops from 230.7 G (re-running the
story) to 3.4 G (cache carried, table exponential, fixed-point attention).
- Turns 0 and 1 verified on GPU (1,834 s and 3,346 s). The site is currently down: its prover hit the 16 GB
  memory cap of its slot and was stopped (the machine was fine).

## 4. Where we are and what's left
1. **Sumcheck at more tokens:** fix the 128-token failure, measure 128 and 256, and find whether and where
   sumcheck's cost crosses 8-bit's.
2. **Full-model sumcheck result** (running) versus 8-bit's 7,677 G and FP32's 809,902 G.
3. **Cut the verification cost** (75% of the sumcheck layer): fewer, larger chunks; a smaller field; or a
   commitment without the big FFT.
4. **Website:** fix its memory budget, restart it, later move it to the unquantized path.
5. Not built yet: one end-to-end Qwen proof (38 segment proofs aggregated into one); in-browser verification.

## 5. Where the code is
| Approach | Repo / branch | Local path |
|---|---|---|
| A. FloatDot + FP32 benchmark | github.com/amiabix/zisk `task/floatdot-transformer-proof-benchmark` | /Users/abix/Work/worktrees/zisk-floatdot-transformer-benchmark |
| B. dot_f8 | github.com/amiabix/zisk `task/dot-f8-sm` | /Users/abix/Work/worktrees/zisk-dot-f8-sm |
| C. Sumcheck | github.com/amiabix/zisk-qwen3-profile `sumcheck-bf16` (private) | /Users/abix/Work/repositories/zisk-qwen3-profile |
| E. Website | not in git yet | /Users/abix/Work/repositories/provable-next-word (runs on Omarchy at ~/pnw-kv/pnw) |

Keep the fork's `task/general-fp32-precompile` (8566a014): this repo pins it as its ziskos dependency.

## 6. Machines and safety
- Mac: development and builds. Omarchy (`ssh omarchy`): Linux, 31 GB RAM, RTX 4080 Laptop 12 GB,
  for local testing only (the target is an RTX 5090, 32 GB). It holds the Qwen3-4B model and the GPU proving key.
- Omarchy crashed twice from running the sumcheck prover and the story GPU prover together. Now:
  benchmarks run via `~/scale-runs/capped.sh NAME CMD` (12 GB `bench.slice`, one at a time, peaks in
  `~/scale-runs/peaks.log`); the website runs only inside the 16 GB `story.slice`. Never bypass this.
- Results live in `~/scale-runs/` (current queue: `run8.sh`, status in `status8`).

## 7. Working rules
Never state facts from memory: read code, cite file:line; measured numbers only, label estimates. No
stubs presented as progress; don't claim done without passing output. Minimal, surgical code in the
existing style; comments only where a first-time reader needs them. Commit/push only when asked; author
Ayush only, no AI attribution. Messages to the ZisK team: terse, facts only.
