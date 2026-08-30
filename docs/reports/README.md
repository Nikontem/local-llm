# Reports

Measurements and the decisions taken from them. These are records of what was observed on a
particular machine on a particular day, kept so a future reader can tell whether the
observations still hold. They are not specifications — the pages one level up in `docs/` are
what describe how the tool behaves.

| Report | What it settles |
| --- | --- |
| [2026-08-30 — two ways of measuring a model's speed](2026-08-30-tuning-engine-comparison.md) | Which of two measurement engines `local-llm tune` ships, and how much the four speed settings are actually worth on a 30B model. |
| [2026-08-29 — llama.cpp against MLX and vllm-metal](2026-08-29-llamacpp-vs-mlx-qwen3-coder.md) | How the three engines compare on the same model and prompts. |
| [2026-08-29 — the MLX decision](2026-08-29-mlx-decision.md) | Why MLX support was not built, and what would reopen the question. |

Every number in these files is a real measurement. Where something was estimated, assumed, or
could not be measured, the report says so.
