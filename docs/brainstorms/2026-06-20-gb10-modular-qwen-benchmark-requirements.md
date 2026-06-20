# GB10 × Modular: Qwen3.6-35B-A3B benchmark & kernel-coverage map

**Date:** 2026-06-20
**Status:** Requirements (ready for planning)
**Scope:** Deep — feature

## Problem & goal

Validate whether the Modular Platform (MAX inference server + Mojo GPU kernels) is
worth running on the GB10 (DGX Spark) for a real workload, and understand *why* by
mapping kernel support. Two reinforcing win conditions:

1. **Beat the incumbent.** Show MAX serves `Qwen3.6-35B-A3B` faster than the current
   llama.cpp / GGUF baseline on this GB10 (tok/s, TTFT, throughput).
2. **Map GB10 kernel support.** Determine which MAX kernels have `sm_121` / `sm_120`
   fast paths vs generic fallback on GB10 — a coverage + perf map that explains the
   scoreboard and exposes optimization gaps worth contributing upstream.

The kernel map gives the scoreboard a diagnosis; the scoreboard gives the map a metric.

## Hardware context (shapes every decision)

- GPU: **NVIDIA GB10**, compute capability **12.1 (`sm_121a`)**, aarch64, CUDA 13.0,
  driver 580. Verified via `nvidia-smi` on this box.
- **128 GB unified LPDDR5X at ~273 GB/s**, shared CPU+GPU. Inference here is
  **memory-bandwidth-bound, not compute-bound** — so a win is about bytes/token and
  quantization efficiency more than tensor-core occupancy.
- This is why the **MoE** (`A3B` = ~3B active params/token) is the compelling case:
  low bytes/token → high tok/s on a bandwidth-limited part.

## Grounding facts (verified in this repo)

- MAX has **first-class DGX Spark support**: `sm_121a` target config and GB10 named
  in `mojo/stdlib/std/gpu/host/info.mojo` (`_get_dgx_spark_target`, `DGXSpark`,
  `version="sm_121"`). This is not a "will it run" gamble.
- MAX supports **Qwen3 incl. MoE**: `max/python/max/pipelines/architectures/qwen3/arch.py`
  registers `Qwen3ForCausalLM` with `example_repo_ids=["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"]`
  and an MoE weight adapter (`convert_qwen3_moe_state_dict`, expert stacking). A newer
  `qwen3_5` arch and `qwen3vl_moe` also exist.
- MAX **loads GGUF K-quants** (`Q4_K`, `Q5_K`, `Q6_K`) — `max/python/max/graph/weights/load_gguf.py:51-53`.
  The baseline file is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (= `Q4_K`), so **MAX can in
  principle load the exact same GGUF llama.cpp runs** → true identical-weights comparison.
- Most optimized kernels (FP8 grouped matmul, structured conv, MMA) target **`sm_100a`**
  (B200) and `sm_90a` (Hopper). Whether they reach `sm_121` or fall back to generic is
  the core goal-2 unknown.

## Local assets (already on disk)

- **Baseline engine:** `~/ard/llama.cpp` (built) + `~/ard/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  (and an `-MTP-` multi-token-prediction variant).
- **MAX inputs (HF checkpoints):** `Qwen--Qwen3.6-35B-A3B` (BF16), `-FP8`,
  `RedHatAI--Qwen3.6-35B-A3B-NVFP4`, plus dense sibling `Qwen--Qwen3.6-27B`.
- **Smoke-test model:** `Qwen--Qwen3-4B-Instruct-2507` (cached) — small, an
  `example_repo_id`-class arch, ideal for rung 0.
- **Comparison points (optional):** `nvidia--Nemotron-3-Nano-Omni-30B-A3B` (BF16/NVFP4 +
  GGUF), `gemma-4-26B-A4B` MoE GGUF.
- **Environment:** Docker present; **no** local Mojo/`max`/`pixi`/bazel toolchain built →
  drives the container-first decision (use prebuilt image, avoid aarch64 source build).

## Chosen approach: container-first de-risk ladder

Each rung gates the next; the cheap failure modes happen first.

- **Rung 0 — Container smoke.** Pull the MAX Docker image (must be an **aarch64** image
  that targets `sm_121`), `max serve` the cached `Qwen3-4B`, hit the OpenAI-compatible
  endpoint, confirm correct generation. Purpose: prove the sm_121 serving path works and
  **resolve the Qwen3.6 config-name risk** before investing in the big model.
- **Rung 1 — Scoreboard.** Serve `Qwen3.6-35B-A3B` on MAX and benchmark vs llama.cpp
  Q4_K_M. Try **identical-GGUF mode first**; fall back to **HF FP8/NVFP4** if MoE+GGUF
  isn't wired. Capture tok/s, TTFT, throughput, memory, with full reproducible metadata.
- **Rung 2 — Explain (kernel map).** Run `kbench` / kernel tests on the MoE-decode ops the
  scoreboard implicates (grouped/expert matmul, attention, RoPE, FP8/FP4 dequant) at
  `sm_121a`; record fast-path vs fallback → the contributable-gap list.
- **Marquee comparison:** MAX-**NVFP4** (Blackwell FP4 tensor cores) vs llama.cpp-Q4. If
  MAX's FP4 path reaches `sm_121`, that's a structural win llama.cpp can't answer; if it's
  `sm_100`-only and falls back, that *is* the top goal-2 finding. Either outcome is a win.

### Fairness modes (decided)

- **Identical-GGUF** = the fair baseline (same `Q4_K_M` weights, two engines) — gold
  standard *if* Qwen3-MoE-from-GGUF runs on GPU at sm_121.
- **Best-effort** = MAX FP8/NVFP4 vs llama.cpp Q4 — each engine's best path; always
  available as fallback and as the "real-world best" framing.

## Success criteria

- **Rung 0:** MAX container serves a local model on GB10 and returns correct output;
  Qwen3.6 config-name dispatch status known (auto-match / needs shim / fall back to 30B-A3B).
- **Rung 1:** A head-to-head table with at least one apples-to-apples row (identical-GGUF
  *or* matched bits-per-weight), reporting MAX vs llama.cpp tok/s, TTFT, throughput — plus
  a clear verdict (faster / slower / parity) and by how much.
- **Rung 2:** A per-op map of sm_121 fast-path vs fallback for the hot MoE-decode kernels,
  naming the specific ops that explain the scoreboard and any worth a contribution.
- **Reproducible:** every result carries full metadata (image tag, model + quant, server
  flags, prompt/seq lengths, warmup) per the `bench-capture` discipline.

## Risks & open questions

- **Qwen3.6 config name.** If the checkpoint's HF `architectures` field isn't a name MAX
  matches, auto-dispatch fails. Mitigation: rung-0 check; fall back to `Qwen3-30B-A3B`
  (already an `example_repo_id`) or a small config shim. *(Resolve in rung 0.)*
- **GGUF + MoE wiring.** The Qwen3 MoE weight adapter targets HF safetensors tensor names;
  GGUF expert layout differs, so identical-GGUF mode may not be wired. *(10-min check; if
  not, use best-effort mode.)*
- **aarch64 image availability.** Need a MAX Docker tag that is ARM64 *and* targets sm_121.
  If none exists, the container-first plan needs a different image or a source build (out
  of current scope). *(Verify at rung 0 pull.)*
- **Quantized GPU kernel coverage on sm_121.** Whether Q4_K / FP4 have GPU kernels at
  sm_121 vs CPU-only dequant is itself a goal-2 finding, not a blocker.

## Explicitly out of scope (for now)

- Local Mojo/bazel **source build** and toolchain setup (container-first avoids it).
- **Training** kernels (rumi / liger / quack / compile) — this is inference-serving focused.
- Non-Qwen models except as rung-0 smoke fillers or optional rung-1 comparison points.
- Multi-GB10 / distributed serving.
