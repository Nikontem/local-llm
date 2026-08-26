# Milestone 2 acceptance — the author's Mac, 2026-08-26

Machine: Apple M4 Pro, 48 GB, macOS 26.5.2, llama-server 0.3.0. Repo at `460f68a`.
All commands run with `uv run local-llm ...` against the real `~/.config/local-llm/models.ini`
and the real Hugging Face Hub. Router stopped beforehand.

## Live test

`uv run pytest tests/live -m live -q` → 1 passed in 10.7 s (25 base models examined;
every use-case group had vendor-lineage candidates with a suggested quantization).

## recommend — first run, and what it exposed

The first run listed the machine line and four groups, but two of the three
"general" picks were `HauhauCS/...Uncensored-Aggressive` and
`LuffyTheFox/...Uncensored-Genesis-Hermes`, both labelled *vendor*, and the
"small" group contained `handy-computer/cohere-transcribe-03-2026-gguf`, a
speech model. Causes and fixes, both data-driven and committed before the
second run:

1. Uploaders of modified models tag their repos `base_model:quantized:<vendor
   model>`, so the tag chain alone cannot see the modification. New rule in
   `hub.lineage`: words in the repo name that the base model's name does not
   contain — beyond packaging vocabulary such as GGUF, imatrix, i1, UD and
   quantization tags — mean the weights were changed; the repo is a
   derivative. Checked against legitimate quantizer naming (`unsloth/X-GGUF`,
   `bartowski/google_gemma-3-12b-it-GGUF`, `mradermacher/X-i1-GGUF`,
   `TheBloke/Llama-2-7B-Chat-GGUF`), all still vendor.
2. GGUF is also used for speech, embedding and image models. `discover` now
   keeps only text pipelines (`text-generation`, `image-text-to-text`,
   `text2text-generation`, or unknown).
3. `antirez/deepseek-v4-gguf` offered a 5.6 GB file for a model whose
   parameter count implies far more — a partial upload. Files under one bit
   per parameter are no longer offered.

Second run (listings served from the 24-hour cache), abridged:

```
Apple M4 Pro, 20 GPU cores, 48 GB unified memory → 38 GB usable for models (10 GB reserved)

coding
   1. Qwen3-Coder-30B-A3B-Instruct  unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  UD-Q4_K_XL  16.5 GB on disk, ~19.9 GB in memory  comfortable  vendor  · in models.ini
general
   2. Qwen3-30B-A3B-Thinking-2507 (thinking)  unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF  UD-Q4_K_XL  ...  comfortable  vendor
   3. Muse-Glimmer-30B (vision)  unsloth/Muse-Glimmer-30B-GGUF  UD-Q4_K_XL  ...  comfortable  vendor
small
   4. Qwen3.5-4B (thinking, vision)  unsloth/Qwen3.5-4B-GGUF  Q8_0  4.8 GB on disk, ~6.5 GB in memory  comfortable  vendor
   5. LFM2.5-2.6B (thinking)  LiquidAI/LFM2.5-2.6B-GGUF  Q8_0  ...
vision
   ...  Qwen3.8-27B (thinking, vision)  unsloth/Qwen3.8-27B-GGUF  UD-Q4_K_XL  17.2 GB on disk, ~20.8 GB in memory  comfortable  vendor  · in models.ini
```

Observations: the quantization suggestion scales with the machine (Q8_0 for
the small models, UD-Q4_K_XL for the 27–30B ones on 38 GB); the two models the
author already runs are marked "in models.ini"; every remaining entry is a
vendor release.

## search

`search "qwen2.5 0.5b" --limit 5` listed five repos with downloads, likes and
lineage, every quantization with size, estimate and fit, and a ready-made
`local-llm pull <repo>:<label>` line. `saidutta69/Qwen2.5-0.5B-Instruct-heretic`
was labelled vendor before the name rule and derivative after it.

## pull → serve → chat → remove

- `pull Qwen/Qwen2.5-0.5B-Instruct-GGUF --yes`: suggested Q8_0 (0.6 GB,
  ~1.7 GB, comfortable), downloaded it, wrote
  `[Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0]` with `c = 32768`, `n-predict = 4096`,
  sampling from the `qwen2.5-small` profile (the repo's model card has no
  numeric recommendations), and the two explanatory comments.
- `models` listed six sections; `up`; `load` printed
  `~2.1 GB (weights + KV cache at c=32768)` and `{"success":true}`.
- Chat request returned `{"content":"Hello!","model":"Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0","tokens":3}`.
- `status` showed the child at 1.1 GB with no "(asleep)" mark — the label now
  comes from the router's own model status.
- `load unsloth/Qwen3.8-27B-GGUF:Q4_K_XL` printed `~22.9 GB (weights + KV cache
  at c=65536)`; `unload` succeeded.
- `remove ... --delete-files --yes` removed the section, the cache symlink and
  its blob; the running router reloaded its list. `down` left no process.
- `models.ini` afterwards differed from the original only by one trailing
  blank line — fixed in `preset.remove_section` (add-then-remove now round-trips
  byte for byte) — and the post-removal hint wrongly suggested loading the
  removed model — fixed.

## Verdict

Every command in the milestone-2 plan works on this machine against the real
Hub. Unit suite: 179 passed, ruff clean; live test passing.
