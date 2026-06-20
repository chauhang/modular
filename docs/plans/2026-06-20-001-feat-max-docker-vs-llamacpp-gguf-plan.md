# feat: MAX Docker vs llama.cpp GGUF serving benchmark on GB10 (first slice)

**Date:** 2026-06-20
**Status:** active
**Depth:** Standard
**Origin:** docs/brainstorms/2026-06-20-gb10-modular-qwen-benchmark-requirements.md

---

## Summary

Stand up the MAX inference server from the prebuilt Docker image on this GB10
(DGX Spark, sm_121a), serve a GGUF model, and produce a first reproducible
head-to-head against the existing llama.cpp baseline using the **same OpenAI
client** (`max benchmark`) against both engines. This is the smallest end-to-end
slice that proves the container → serve → benchmark pipeline and puts a real
tok/s / TTFT number on the board. The kernel-coverage map and the GPU-vs-GPU
(bf16) comparison are explicitly deferred to follow-up slices.

---

## Problem Frame

The origin brainstorm scopes two goals (beat llama.cpp; map sm_121 kernel
coverage). Research (see origin + below) showed the kernel reality is harsher
than assumed: MAX's optimized kernels are gated to `_is_sm10x_gpu()` (B100/B200/
B300) or H100, so GB10 falls to generic/naive/CPU paths, and `q4_k` GGUF is
**CPU-only** in MAX. Rather than chase a full Deep effort up front, this plan
takes the user's redirect — **simplest test first** — to de-risk the pipeline
and generate the first honest data point before investing in the kernel map.

This first comparison is **lopsided by construction**: MAX serves the Q4_K GGUF
on **CPU** while llama.cpp serves the identical file on **GPU**. That is
acceptable and even useful here — it proves the pipeline, empirically confirms
the CPU-only finding, and captures llama.cpp's GPU number. It is *not* the
GPU-vs-GPU verdict (that is the deferred bf16 slice).

---

## Requirements

- **R1.** Pull and run the MAX server from the prebuilt Docker image on GB10
  (aarch64), confirming an image tag that exists for this architecture.
  *(origin: goal "prove it runs" precursor)*
- **R2.** Serve a GGUF model through MAX's OpenAI-compatible endpoint and get a
  correct generation. *(origin: rung 0)*
- **R3.** Run the same `max benchmark` client against both the MAX server and the
  llama.cpp server on the **same GGUF file**, reporting tok/s, TTFT, TPOT, and
  request latency percentiles. *(origin: rung 1 scoreboard, narrowed)*
- **R4.** Capture results with full reproducible metadata (image tag, engine
  versions, model + quant, device/placement, seq/bs, warmup/timed iters,
  timestamp, git hash) per the `bench-capture` discipline, saved to the repo. *(origin: success criteria — reproducible)*
- **R5.** Record the verdict honestly, including the CPU-vs-GPU asymmetry and
  which device MAX actually used for the GGUF. *(origin: fairness modes)*

---

## Key Technical Decisions

- **KTD1 — Same GGUF, same client.** Use the existing
  `~/ard/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` for both engines and drive both
  with `max benchmark` (`--backend modular` for MAX, an OpenAI driver against
  llama.cpp's `llama-server`). One load generator eliminates client-side
  measurement skew. *(research: `max/python/max/benchmark/benchmark_serving.py` hits any OpenAI endpoint and reports tok/s, TTFT, TPOT, ITL, percentiles)*
- **KTD2 — Accept MAX-on-CPU for this slice.** `q4_k` is CPU-only in MAX's
  `SupportedEncoding`. We do **not** fight this here; we measure it and report
  the device used. The GPU-vs-GPU comparison (MAX bf16) is a separate deferred
  slice. *(research: SupportedEncoding marks q4_k/q4_0/q6_k CPU-only)*
- **KTD3 — Smoke with a small model first.** Before the 35B run, validate the
  container + serve + benchmark loop on a small cached model
  (`Qwen3-4B-Instruct-2507` or `modularai/Llama-3.1-8B-Instruct-GGUF`) to keep
  failures cheap and fast. *(origin: de-risk ladder rung 0)*
- **KTD4 — GB10 OOM-hang pre-flight.** Before any 35B run, verify memory
  headroom and the agreed memguard/earlyOOM threshold — GPU memory is invisible
  to earlyOOM on this box and a too-large run hangs it. *(bench-capture: pre-flight memory check)*
- **KTD5 — Serve flags.** MAX: `max serve --weight-path <gguf> --devices <cpu|gpu:0> --port 8000`; arch auto-detected (sm_121), `--target cuda:sm_121` only if override needed. llama.cpp: `llama-server -m <gguf> -ngl 99 --host 127.0.0.1 --port 8001`. *(research: serving agent flag map)*

---

## Scope Boundaries

### In scope
- Container bring-up, single-GGUF serving on both engines, one shared-client
  benchmark comparison, reproducible metadata capture, honest verdict.

### Deferred to Follow-Up Work
- **GPU-vs-GPU comparison** — MAX `bfloat16` (HF checkpoint) vs llama.cpp Q4_K_M,
  the real on-GPU verdict. Next slice after this one lands.
- **Kernel-coverage map (goal 2)** — empirically confirm the source-traced
  sm_121 fallback map (grouped matmul, MLA, FP8/FP4) via `kbench`. Large enough
  for its own plan.
- **fp8 / fp4 "does it start" probes** — likely blocked by the SM100 `comptime
  assert`; worth a one-shot probe but not in this slice.
- **Contribution spike** — relaxing an arch guard (e.g. grouped-matmul or the
  FP4 assert) to add sm_120/sm_121, then re-measuring.
- **MTP / dense-27B / Nemotron / gemma-MoE comparison points.**

### Out of scope (this product effort)
- Local Mojo/bazel source build and toolchain setup (container-first).
- Training kernels (rumi / liger / quack), multi-GB10 / distributed serving.

---

## Implementation Units

### U1. Verify and pull the MAX Docker image for aarch64/GB10
**Goal:** Confirm a MAX server image tag that exists for linux/arm64 and pull it;
confirm it starts and sees the GB10 GPU.
**Requirements:** R1
**Dependencies:** none
**Files:**
- `docs/benchmarks/gb10-max-vs-llamacpp/README.md` (new — record exact image tag, digest, `docker run` invocation)
**Approach:** Resolve the right tag for `docker.modular.com/modular/max-nvidia-full`
on arm64 (the `:latest` multi-arch manifest may or may not include arm64; check
`docker manifest inspect` before pulling). Run with `--gpus=1`, mount
`~/.cache/huggingface` and `~/ard/models`, and confirm `nvidia-smi` inside the
container reports GB10. If no arm64 image exists, that is a hard finding — record
it and stop the container path (flagged as the top risk).
**Patterns to follow:** `docker run` example in `CLAUDE.md` (MAX Server Commands).
**Test scenarios:**
- Image manifest includes `linux/arm64`; pull succeeds (record digest).
- Container starts and `nvidia-smi` inside it lists `NVIDIA GB10`.
- Failure path: if arm64 absent, README records the gap and the plan halts here.
**Verification:** Container runs and the GPU is visible from inside it.

### U2. Smoke test — serve a small GGUF and validate the benchmark loop
**Goal:** Prove serve + OpenAI endpoint + `max benchmark` end-to-end on a small
model before the 35B run.
**Requirements:** R2, R3 (small-scale)
**Dependencies:** U1
**Files:**
- `docs/benchmarks/gb10-max-vs-llamacpp/smoke.md` (new — commands + first output)
**Approach:** `max serve` a small cached model
(`modularai/Llama-3.1-8B-Instruct-GGUF` or `Qwen3-4B-Instruct-2507`), curl
`/v1/chat/completions` for a correct generation, then run `max benchmark
--backend modular` against it for a handful of prompts. Confirm metrics parse.
Resolves the Qwen3.6 config-name risk if a Qwen3 model is used.
**Patterns to follow:** `max/python/max/benchmark/benchmark_serving.py` invocation;
serving flag map from research.
**Test scenarios:**
- Server returns a coherent completion for a fixed prompt.
- `max benchmark` completes and emits tok/s + TTFT for the small model.
- Endpoint reachable on the chosen port; clean shutdown frees the GPU/CPU.
**Verification:** A small-model benchmark JSON/console report is produced and sane.

### U3. Head-to-head — Qwen3.6-35B-A3B Q4_K_M on MAX vs llama.cpp
**Goal:** Serve the same Q4_K_M GGUF on both engines and benchmark with the same
client; capture the device MAX actually used.
**Requirements:** R3, R5
**Dependencies:** U2, U4 (capture harness ready)
**Files:**
- `docs/benchmarks/gb10-max-vs-llamacpp/run.md` (new — exact commands per engine)
**Approach:** Run the OOM-hang pre-flight (KTD4). Start MAX with
`--weight-path ~/ard/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (expect CPU
placement for q4_k; record what `--devices` resolves to). Start `llama-server`
with the same file and `-ngl 99` (GPU). Run `max benchmark` against each with an
**identical** prompt set, output length, warmup, and concurrency. Run engines
**sequentially**, never concurrently.
**Patterns to follow:** `bench-capture` skill — sequential runs, identical warmup.
**Test scenarios:**
- Both servers serve the same GGUF and return coherent output.
- Benchmark run uses identical prompts/output-length/warmup/concurrency across both.
- Record which device MAX used (confirm CPU for q4_k) and llama.cpp's `-ngl` offload.
- Negative-result-is-valid: a MAX loss (CPU vs GPU) is recorded as a finding, not a failure.
**Verification:** Two comparable benchmark result sets exist for the 35B model.

### U4. Capture results and write the honest verdict
**Goal:** Persist results with full metadata and a verdict that states the
CPU-vs-GPU asymmetry plainly.
**Requirements:** R4, R5
**Dependencies:** U2 (harness shape), feeds U3
**Files:**
- `docs/benchmarks/gb10-max-vs-llamacpp/results/` (new — per-run JSON)
- `docs/benchmarks/gb10-max-vs-llamacpp/REPORT.md` (new — verdict + metadata table)
- `docs/benchmarks/gb10-max-vs-llamacpp/status.md` (new — incremental status)
**Approach:** Result JSON carries timestamp (generated in-harness), git hash,
MAX image tag/digest, llama.cpp commit, model + quant, device used, seq/bs,
warmup/timed iters, GPU name, and the metrics. REPORT.md tabulates MAX-CPU vs
llama.cpp-GPU and links the next deferred slice (bf16 GPU comparison).
**Patterns to follow:** `bench-capture` metadata contract; save to repo not `/tmp`.
**Test scenarios:**
- Each result JSON contains every required metadata field (assert presence).
- REPORT.md and status.md agree (no "Running" vs "BLOCKED" mismatch).
- No number is carried forward unlabeled; CPU/GPU device is stated per row.
- Test expectation: lightweight schema/consistency assertions on the JSON, not unit tests of MAX.
**Verification:** REPORT.md gives a one-glance verdict with reproducible metadata.

---

## Risks & Dependencies

- **No arm64 MAX image** (top risk). If `docker.modular.com/modular/max-nvidia-full`
  has no `linux/arm64` manifest, the container-first path is blocked → fall back
  to a source/pixi build (out of scope) or report the gap. *Resolved in U1.*
- **q4_k CPU performance on 35B.** A 35B MoE on CPU may be very slow; set a
  generous benchmark timeout and small prompt count for the MAX-CPU run.
- **OOM-hang on GB10.** Mitigated by KTD4 pre-flight; GPU memory invisible to
  earlyOOM.
- **Qwen3.6 config-name dispatch.** If MAX doesn't auto-match Qwen3.6, U2's
  Qwen3 smoke surfaces it early; fall back to `Qwen3-30B-A3B` mapping or a shim.
- **Dependency:** `~/ard/llama.cpp/build/bin/llama-server` (present), cached GGUF
  (present), Docker (present).

---

## Sources & Research

- **Origin:** docs/brainstorms/2026-06-20-gb10-modular-qwen-benchmark-requirements.md
- **Serving surface (repo research):** `max serve` flags (`--weight-path`,
  `--quantization-encoding`, `--devices`, `--port`); `SupportedEncoding` marks
  `q4_k/q4_0/q6_k` **CPU-only**, GPU encodings = bf16/fp16/fp8_e4m3fn/
  float4_e2m1fnx2/gptq; `max benchmark` client at
  `max/python/max/benchmark/benchmark_serving.py` (tok/s, TTFT, TPOT, ITL,
  percentiles; `--backend modular|vllm|sglang|trtllm`); image
  `docker.modular.com/modular/max-nvidia-full:latest`.
- **Kernel arch dispatch (repo research):** optimized kernels gated to
  `_is_sm10x_gpu()` (B100/B200/B300) or H100; GB10 (sm_121/120) falls to
  naive grouped-matmul (`grouped_matmul.mojo:1174`), cuBLAS dense matmul, no
  MLA path; FP8/FP4 quant `comptime assert _is_sm10x_gpu` (SM100-only). Feeds the
  deferred kernel-map slice.
- **GB10 facts (verified on box):** `nvidia-smi` → NVIDIA GB10, sm_121 (cc 12.1),
  aarch64, CUDA 13.0; 128GB unified LPDDR5X ~273 GB/s (bandwidth-bound).
- **Discipline:** `bench-capture` skill (sequential runs, identical warmup,
  in-script metadata, repo-saved results, incremental status).
