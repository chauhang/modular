# MAX on GB10 (DGX Spark, sm_121) — findings

Goal: serve `Qwen3.6-35B-A3B` (MoE) on MAX vs the llama.cpp Q4 baseline, and map
which MAX paths actually work on GB10. What we found is that the blocker is
*model-architecture and kernel support*, not raw performance — so this report is
primarily the support/coverage map (goal 2), with the head-to-head gated behind
getting MAX to run the model at all.

Hardware: NVIDIA GB10, sm_121 (cc 12.1), aarch64, 128 GB unified LPDDR5X
(~273 GB/s), CUDA 13.0, driver 580.

## TL;DR

- **llama.cpp baseline (the number to beat):** Q4_K_M, GPU, all layers offloaded
  → **68.6 tok/s generation**, 2332 tok/s prefill (20.6 GiB, 34.66 B params).
- **MAX runs on GB10's GPU** — but only via **bf16/fp16** for *supported*
  architectures (proven: Qwen2.5-0.5B serves + generates on GPU).
- **`Qwen3.6-35B-A3B` (MoE) is not supported** by the `:latest` MAX image:
  `MAX-optimized architecture not available`. The **dense** `Qwen3.6-27B` *is*
  supported — so the blocker is the **MoE variant**, not Qwen3.6 generally.
- **No GPU low-bit path on GB10:** GGUF K-quants are CPU-only by config; FP8/FP4
  kernels are hard-gated to SM100 (B200) and don't cover sm_121. MAX's only GB10
  GPU precision is bf16 — structurally ~3× the bytes/token of llama.cpp's Q4 on a
  bandwidth-bound box.
- **Nightly image does NOT fix it:** `:nightly` also rejects the MoE with a
  clearer error — `Architecture 'Qwen3_5MoeForConditionalGeneration' not found in
  registry`. The missing arch is **`qwen3_5_moe`**; the dense `qwen3_5` works.

## 0. Serving throughput measured on GB10

| Engine | Model | Precision | Device | Gen tok/s | TTFT | Notes |
|---|---|---|---|---|---|---|
| llama.cpp | Qwen3.6-35B-A3B (MoE) | Q4_K_M | GPU | **68.6** | — | 20.6 GiB, ~3B active/token |
| MAX | gemma-4-31B-it (dense) | bf16 | GPU | **1.94** | 0.86 s | 58 GiB; 450 s cold compile |

These are **different models** (the only ones each engine could run) — not a direct
head-to-head. But the contrast is the story: MAX on GB10 is forced to bf16 (no GPU
low-bit path), so a 31B dense model reads ~58 GB/token and lands at 1.94 tok/s —
~40% of the 273 GB/s bandwidth ceiling (~4.7 tok/s theoretical). llama.cpp's MoE
at Q4 reads a fraction of that per token. A true same-model head-to-head needs a
MAX-supported model with a matching GGUF (see matched-pair note).

## 1. llama.cpp baseline (GB10 GPU)

```
llama-bench -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -p 512 -n 128
| qwen35moe 35B.A3B Q4_K - Medium | 20.60 GiB | 34.66 B | CUDA | 99 | pp512 | 2331.80 ± 21.73 |
| qwen35moe 35B.A3B Q4_K - Medium | 20.60 GiB | 34.66 B | CUDA | 99 | tg128 |   68.57 ± 0.31 |
```

## 2. MAX encoding × device support (source: `config_enums.py:83-91`)

| Encoding | CPU | GPU | Notes |
|---|---|---|---|
| float32 | ✓ | ✓ | |
| float16 / bfloat16 | ✗ | ✓ | **the working GB10 GPU path** |
| float8_e4m3fn (FP8) | ✗ | ✓* | *GPU-listed, but kernels SM100-gated (see §4) |
| float4_e2m1fnx2 (NVFP4) | ✗ | ✓* | *GPU-listed, but kernels SM100-gated (see §4) |
| q4_k / q4_0 / q6_k (GGUF) | ✓ | ✗ | **CPU-only** — can't be the GPU comparison |
| gptq | ✗ | ✓ | not tested (no local GPTQ checkpoint) |

## 3. Model architecture support — `:latest` image (empirical sweep)

| Model | HF architecture / model_type | `:latest` | `:nightly` |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | `Qwen2ForCausalLM` | ✓ serves+generates | — |
| Qwen3-4B-Instruct-2507 | `Qwen3ForCausalLM` | ✓ supported | — |
| **Qwen3.6-27B** (dense) | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | **✓ supported** | — |
| **Qwen3.6-35B-A3B** (MoE) | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | **✗ unsupported** | **✗ unsupported** |

→ The blocker is precisely the **`qwen3_5_moe`** architecture. The dense `qwen3_5`
sibling is supported; the MoE is not, in **either** the `:latest` or `:nightly`
image. All four precisions of `Qwen3.6-35B-A3B` (bf16, FP8, NVFP4, GGUF) fail at the
same architecture-registry gate — precision never gets a chance to matter.

**For a `modul.ar/request`:** add `Qwen3_5MoeForConditionalGeneration`
(`model_type: qwen3_5_moe`) to the MAX pipeline registry.

### Full local-model inventory (GB10, `:latest`/`:nightly`)

| Model | Arch | Result | Local GGUF? |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Qwen2 | ✅ serves | no |
| Qwen3-4B-Instruct-2507 / -Base | Qwen3 | ✅ supported | no |
| Qwen3Guard-Gen-0.6B | Qwen3 | ✅ supported | no |
| **Qwen3.6-27B** (dense) | qwen3_5 | ✅ supported | no |
| Llama-3.2-1B-Instruct | Llama3 | ✅ supported | no |
| gemma-4-31B-it | gemma4 | ✅ **SERVES** (bf16 GPU, ready 440s) — see note | no |
| gemma-4-E2B-it | gemma4 | ⚠️ arch ok, code bug (`% NoneType`) on E2B MatFormer | no |
| **Qwen3.6-35B-A3B** (+FP8/NVFP4) | qwen3_5_moe | ❌ arch not in registry | **Q4_K** |
| **Nemotron-3-Nano-30B-A3B** | NemotronH_Nano_Omni_V3 | ❌ arch not in registry | **Q4_K / MXFP4** |
| gemma-2-2b-it | gemma2 | ❌ arch not in registry | no |
| gemma-4-26B-A4B (MoE) | gemma4 | not tested (no local HF; GGUF only) | **Q4_K_XL** |

**gemma-4-31B-it note:** serves on GB10 GPU in bf16, but only with modest settings
(`--max-length 8192 --max-batch-size 1 --device-memory-utilization 0.95`). The
user's `--max-length 32768 --max-batch-size 8` OOMs the memory estimator
(`model 58.25 GiB + activations 15 GiB don't leave room for KV cache`). Even when
it serves, the 58 GiB bf16 weights leave only **1.72 GiB for KV cache** — practical
proof that GB10's lack of a GPU low-bit path forces large bf16 footprints that
starve the KV cache. (Also note: `--enforce-eager`/`--max-model-len` are vLLM
flags, not MAX; MAX uses `--max-length` and is eager-compiled differently.)

### Matched-pair conclusion

A real head-to-head needs a model with **both** a MAX-supported HF checkpoint and a
local GGUF. **None of the fully-local pairs qualify:** every local GGUF is an MoE
MAX can't serve (Qwen3.6-A3B, Nemotron-A3B, gemma-4-26B-A4B), and every
MAX-supported local model is dense with no local GGUF. So a head-to-head requires
**one download**:

- **Smallest/fastest:** a GGUF for a supported dense local model (Llama-3.2-1B or
  Qwen3-4B) → immediate matched comparison, but small + dense (not the MoE story).
- **Keeps the MoE story:** download `Qwen3-30B-A3B` (`Qwen3MoeForCausalLM` — *is*
  registered) + its GGUF → finally tests MoE serving and the sm_121 naive
  grouped-matmul fallback vs llama.cpp.
- **Reuses an existing GGUF:** download the HF safetensors for `gemma-4-26B-A4B`
  (we already hold its Q4_K_XL GGUF); viable **iff** MAX's gemma4 path handles the
  A4B MoE — untested.

## 4. Kernel arch coverage for GB10 (source trace)

Optimized kernels gate on `_is_sm10x_gpu()` (B100/B200/B300) or H100 (sm_90a).
GB10 = sm_121 is **not** covered:

| Kernel (MoE-relevant) | GB10 behavior |
|---|---|
| Grouped/expert matmul (MoE core) | naive fallback (`grouped_matmul.mojo:1174`) |
| Dense matmul | generic cuBLAS fallback |
| MLA / MHA attention | no consumer-Blackwell path |
| FP8 quant | `comptime assert _is_sm10x_gpu` → fails off SM100 |
| FP4 / NVFP4 quant | same SM100-only assert (`fp4_quantization.mojo:140`) |
| RoPE | portable ✓ |

## 4b. Mojo kernel benchmarks on GB10 (bazel)

Built/run from source with `./bazelw run //max/kernels/benchmarks:<target>` (the
container ships mojo 1.0.0b3.dev2026062006; `bazelw` builds locally on aarch64).
The benches themselves encode the coverage map via `target_compatible_with`:

**Runnable on GB10** (generic NVIDIA, gated only against Apple):
`bench_matmul` (dense), `bench_grouped_matmul` (MoE expert matmul), `bench_mha`,
`bench_mla`, `bench_moe_routing`, `bench_bmm`.

**Incompatible on GB10** (`//:b200_gpu` only — won't build for sm_121):
`bench_blockwise_fp8_1d2d`, `bench_block_scaled_matmul`, `bench_mma_throughput_sm100`,
`bench_matmul_tma_epilogue`, `bench_matmul_1d_tma_epilogue`, `bench_conv2d_sm100`,
`profile_grouped_matmul_swiglu_nvfp4`, `bench_cp_async_bulk`. The optimized
Blackwell microbenchmarks are literally unbuildable on GB10 — bazel-level
confirmation of the SM100 gating found in §4.

Measured numbers (sm_121, GB10):

| Kernel bench | Shape | Result |
|---|---|---|
| `bench_matmul` bf16 | 1024 × 16384 × 512 (transpose_b) | 0.305 ms → **56.4 TFLOP/s** |
| `bench_grouped_matmul` (MoE) bf16 | 1×256×256, 1 expert (default) | 0.174 ms → 192.6 GFLOP/s |

Both build and run on sm_121 (`TEST PASSED`, EXIT 0; first build 236 s incl.
toolchain, then ~10 s cached). The dense path at ~56 TFLOP/s bf16 shows the generic
NVIDIA matmul is functional and performant on GB10 — the llama.cpp gap is the
serving-level precision/footprint problem (§0), not raw dense-matmul throughput.
The grouped (MoE) kernel runs via the generic path too.

**Shape note:** the per-bench `N/K/num_experts` are **comptime** defines baked at
build; `bazel run -- get_defined_int[N]=…` does **not** override them (still ran
256³/1-expert). Realistic-shape MoE sweeps require rebuilding via the autotune
harness `max/kernels/benchmarks/autotune/kbench.py` with a YAML — available, not run
this round. The default-shape numbers above are real but not representative of
Qwen-A3B expert dimensions.

## 5. External corroboration

Spheron, *"Modular MAX + Mojo GPU Cloud LLM Inference"*
(https://www.spheron.network/blog/modular-max-mojo-gpu-cloud-llm-inference/):

- Supported NVIDIA GPUs listed: H100, H200, B200, B300, GB200, GB300, RTX PRO 6000.
  **No GB10 / DGX Spark / sm_121.** (RTX PRO 6000 = sm_120 workstation; sm_121 is a
  distinct, unlisted target.)
- *"MAX's support for mixture-of-experts models is still maturing as of May 2026"* —
  MoE archs (Llama 4 Maverick, DeepSeek V3) "not fully optimized." Directly
  corroborates §3 (MoE blocked) and §4 (naive grouped-matmul).
- Production quant = FP8/FP16 + fp8 KV cache; no NVFP4/GGUF. Matches §2.
- Their MAX-vs-vLLM win (2150 vs 1850 tok/s) is on **H100**, not GB10.

Three independent signals — the runtime `architecture not available` error, the
source-traced SM100 kernel gates, and the vendor writeup — converge: **MAX + GB10 +
Qwen3.6-MoE is not a supported path today.**

**Independent kernel-quality validation (ORNL, SC'25).** Godoy et al.,
*"Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs"*
(arXiv:2509.21039), benchmarked Mojo vs vendor CUDA/HIP on H100 + MI300A: Mojo is
**on par with CUDA/HIP for memory-bound kernels with zero hand-tuning** (Φ̄ ≈
0.92–0.96; BabelStream Triad 1.01×), with gaps on compute-bound (no fast-math, more
registers). Since GB10 inference is memory-bandwidth-bound, this says Mojo's kernel
quality is *not* the GB10 bottleneck — the footprint/quant path is. See
`KERNEL_APPROACHES.md` for the full kernel comparison (Mojo vs CUDA/Triton/ThunderKittens).

## 6. Reproduction

- `probe_max_paths.sh phaseA|phaseB` (set `MAX_IMAGE=…:nightly` to switch image)
- `arch_sweep.sh` — local-model arch-registration sweep
- Results: `results/*.tsv` (per-image archived with `_latest-image` / nightly suffix)
