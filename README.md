# qproof

Proof Carrying Inference for Qwen3-4B on the ZisK zkVM. The harness uses the original BF16 weights and the `dot_f8` precompile to check inference against a pinned Hugging Face reference.

The 32-token prompt with eight predictions matched all 37 activation boundaries and every logit in native execution. On the 500-token prompt with 250 predictions, all greedy token IDs matched, but decode scores diverged. Full-model GPU proof generation remains unverified.

- [Harness architecture and request verification](FULL_MODEL_HARNESS.md)
- [Reference configuration and numerical parity](harness/measurements/QWEN_BF16_PARITY.md)
- [Guest execution costs](harness/measurements/QWEN_FULL_MODEL_COST_1X1.md)
- [Build the client and run the preflight](harness/measurements/RENTAL_READINESS.md)
