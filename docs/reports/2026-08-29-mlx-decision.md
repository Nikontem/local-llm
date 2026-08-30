# MLX support — decision

**Date:** 2026-08-29
**Decision:** MLX support is not being built. `local-llm` stays a single-engine tool
around `llama-server`.
**Reopen when:** one of the conditions in the last section is met.

## What was asked

Whether `local-llm` should serve MLX models — the Apple-Silicon-native model format —
alongside the GGUF models it serves through `llama-server`, behind one endpoint, with
the same per-model tuning, on Apple Silicon only.

## What it would have cost

Whichever MLX engine was chosen, the tool would have gained: a Go reverse proxy
(llama-swap) from a third-party Homebrew tap as the front door; a second Python
environment with its own version pins; a second model format whose downloads are
directories rather than files; a second discovery path on Hugging Face; a second memory
sizing formula; a second spawn-command generator; two runtime topologies for `up`,
`status` and `doctor` to understand and test; and a dependency on projects that ship
breaking changes weekly. Behaviour could not have been uniform across the two formats:
several tuning keys (`min-p`, `presence-penalty`, KV-cache quantization) exist for one
engine only, memory is evict-by-count on one side and reserve-at-spawn on the other, and
switching model is seconds on one side and a cold process start on the other. The
detail is in `vllm-metal-assessment.md`.

## What it would have bought

Measured on the target machine (Apple M4 Pro, 48 GB) with the model the tool actually
serves, Qwen3-Coder-30B-A3B at 4-bit, on a 20.8k-token prompt — the shape of a coding
agent's turn. Full method and numbers in `benchmark-qwen3-coder.md`.

| Served through | Prefill tok/s | Generation tok/s | Time to first token | Physical memory |
|---|---|---|---|---|
| `llama-server` (today) | 338 | 37.9 | 61.8 s | 9.0 GB |
| `mlx_lm.server` | 412 | 31.8 | 50.5 s | 19.9 GB, growing ~2 GB per conversation |
| `vllm-metal` | 342 | 21.9 | 60.9 s | 33 GB, swapped |

MLX's engine is genuinely faster in isolation (35% prefill, 17% generation from the
command line), but through its HTTP server the generation advantage disappears and a
22% prefill edge remains — about ten seconds on a 20k-token turn. Against that: double
the physical memory because MLX loads weights into arrays where llama.cpp maps the file;
a per-conversation KV cache in `mlx_lm.server` that is never evicted; a streaming bug
that delivered a whole answer in `delta.reasoning` instead of `delta.content`; and,
for `mlx_vlm.server`, sampling parameters that have no effect (its issue #1818,
reproduced). `vllm-metal` was slowest on every metric and could not start at all with a
memory fraction that left room for anything else.

Third-party benchmark evidence was also collected and mostly discarded: the only two
verifiable same-machine comparisons found show that most of the "MLX is much faster"
folklore is MLX versus Ollama's wrapper rather than versus `llama-server`, and that
`llama.cpp` with flash attention is roughly 2× ahead of MLX on decode at 100k+ tokens.

## Things learned that stand on their own

- Claude Code must authenticate with a Bearer token to reach anything except
  `llama-server` itself: llama-swap forwards the client's header unchanged, and both
  vLLM and `mlx_vlm.server` reject `x-api-key`. If a proxy or second engine is ever put
  in front, `agents.claude_env` switches from `ANTHROPIC_API_KEY` to
  `ANTHROPIC_AUTH_TOKEN`. `llama-server` accepts both, so nothing changes today.
- llama-swap does no Anthropic-to-OpenAI translation; `/v1/messages` is routed on the
  `model` field and passed through byte-for-byte. Any upstream behind it must serve
  `/v1/messages` natively. `llama-server` has since late 2025; vLLM since 0.11.1.
- Both `llama-server` and `mlx_lm.server` cache prompts. A benchmark that repeats an
  identical prompt measures the cache, not the engine.
- `llama-server` serving a mixture-of-experts model has a physical footprint far below
  the file size, because only the experts each token touches are paged in. Memory
  comparisons with array-loading engines must use physical footprint, not file size.

## When to reopen

- **A machine where memory is not the constraint and prompts dominate**: an Ultra-class
  Mac Studio with 128 GB or more running a large dense model, where MLX's prefill
  advantage is worth more and its memory penalty is worth less.
- **The M5 generation or later**, whose GPU matrix accelerators MLX exploits first;
  Apple's own numbers claim several times faster time-to-first-token. If `llama.cpp`'s
  Metal backend lags on that hardware, re-measure with the harness in the scratch
  scripts described in `benchmark-qwen3-coder.md`.
- **A shared machine serving several developers at once**, where continuous batching
  and a paged KV cache matter. That is `vllm-metal`'s design target — but it is a
  different product from this tool.
- **Either MLX server gaining memory-mapped weight loading and KV-cache eviction**, which
  would remove the two measured penalties.

None of these describes an M4 Pro laptop, so the decision holds for the machine the
tool is developed on.

## Files in this directory

- `decision.md` — this file, the entry point.
- `benchmark-qwen3-coder.md` — the measurement, method, commands, and versions.
- `vllm-metal-assessment.md` — knob-by-knob audit of `vllm-metal` against the tool's
  tuning, with mitigations, plus the Claude Code authentication findings.
- `research-handoff.md` — the first session's research into `mlx_lm.server`,
  `mlx_vlm.server` and llama-swap; background only.
