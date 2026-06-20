# Kernel approaches on GB10: Mojo vs CUDA / Triton / ThunderKittens

Companion to `FINDINGS.md`. Compares how the GB10-compatible kernels are written
and how Mojo's approach differs, with the compile verified as GB10-native and
external (third-party) validation of Mojo kernel quality.

## 1. The kernel benches were compiled from source, GB10-native

The Mojo kernel benches were **built from source on the host via `./bazelw run`**
(not the Docker container — that was only used for serving):

- `bazelw` fetched a hermetic Mojo toolchain (`rules_mojo … mojo_toolchain_linux_aarch64/bin/mojo`)
  and produced a **native aarch64 ELF** at `bazel-bin/max/kernels/benchmarks/gpu/linalg/bench_matmul`.
- **Target verified correct for GB10.** bazel auto-detected the GPU and mapped it
  via `bazel/common.MODULE.bazel:414` → `"gb10": "nvidia:sm_121"`, distinct from
  RTX-50 (`sm_120a`). The binary contains 16× `sm_121a` + `ptx88`, matching the
  DGX Spark target in `mojo/stdlib/std/gpu/host/info.mojo:757`
  (`arch="sm_121a"`, `features="+ptx88,+sm_121a"`, `tune_cpu="sm_121a"`).
- So: **correct sm_121a codegen, no missed GB10 flag.** The ceiling is the
  *algorithm*, not the compile (see §3).

## 2. Mojo runs its OWN machinery on GB10 — no CUDA/cuBLAS fallback

A correction to an earlier assumption: on GB10 the kernels do **not** fall back to
cuBLAS. The matmul dispatch (`linalg/matmul/gpu/__init__.mojo`) gates the optimized
paths on `_has_blackwell_tcgen05()` (= `SM_100X|SM_101X|SM_103X`, which **excludes**
sm_121) and `== H100`. GB10 matches neither, so it routes to Mojo's **generic
`multistage_gemm`** kernel. `bench_matmul` defaults to `use_vendor_blas=False`, so
the measured **56.4 TFLOP/s bf16 is Mojo's own kernel**, not cuBLAS. Grouped/MoE
matmul similarly routes to Mojo's `naive_grouped_matmul`.

→ On GB10 it's **generic Mojo kernels vs arch-specialized Mojo kernels** (the
sm_100 tcgen05 / sm_90 wgmma paths are gated off), not Mojo vs CUDA.

## 3. How the GB10-compatible kernels are written — four approaches

| | Language | Tile/layout abstraction | Multi-arch model | GB10 / sm_121 today |
|---|---|---|---|---|
| **kernels-community** (HF Hub) | 1,632 CUDA files + 617 Triton + 29 Metal | per-kernel hand-rolled; CuTe in newest | hand-written per-arch (**flash-attn3 = 451 `.cu` files**) OR Triton JIT | CUDA mostly sm_80/90/100; **Triton JITs to sm_121** (untuned) |
| **ThunderKittens** | C++ header DSL in CUDA | fixed 16×16 register/shared tiles, hardware-mapped | AOT C++ templates, per-arch macros (`-DKITTENS_SM90/100`) | **upstream: NO GB10.** Our team added an sm_120/121 WGMMA→mma.sync shim for **3 kernels** (GEMM, MHA, RoPE), bf16, untuned |
| **Mojo / MAX** | one language, multi-target (MLIR/KGEN→PTX) | parameterized **layout algebra**, type-safe | **one source, comptime multi-arch dispatch** (+AMD/Apple/CPU) | **own generic Mojo kernels** (`multistage_gemm` 56.4 TFLOP/s; naive grouped) |
| Triton | Python JIT | software tiles (no HW mapping) | runtime JIT, backend-agnostic | JITs to sm_121 automatically, untuned |

### What is genuinely special about Mojo (survives scrutiny)

- **One source → every architecture.** flash-attn3 ships **451 generated `.cu`
  files** (per head_dim×dtype×sm); flash-attn2/sage-attention/flash-mla hand-write
  per-SM trees. Mojo's `grouped_matmul.mojo` / `matmul/gpu` / `mla.mojo` dispatch
  sm_90/sm_100/AMD/Apple from one source via `comptime if`. Adding an arch is a
  *branch*, not a file-tree.
- **Comptime metaprogramming = CUTLASS templates + Triton `constexpr`, unified.**
  Compile-time tile/dtype/shape specialization from normal-looking code, with type
  safety, no Python JIT.
- **The `layout` package** — composable, type-safe tile/swizzle/layout algebra
  reused across matmul and attention. Tellingly, the *newest* community kernels
  (flash-attn4, sonic-moe, quack) are migrating to CUTLASS's **CuTe DSL** —
  convergent evolution toward what Mojo's layout package already is.
- **Type safety into GPU code** — mutability, address space, pointer lifetime, and
  layout in the type system (`TileTensor[mut, dtype, LayoutType, origin, address_space]`).
- **Only approach that is single-source AND multi-vendor** (NVIDIA + AMD + Apple +
  CPU). ThunderKittens is NVIDIA-only; community kernels are per-backend silos.

### Where Mojo currently loses on GB10

- No sm_121/sm_120 *specialized* algorithm — it runs generic kernels. Triton
  auto-JITs to sm_121; our TK shim deliberately targets sm_120/121. Neither is
  tuned, but both put intent into consumer Blackwell that MAX hasn't yet.
- Adding an sm_121 path in Mojo is a comptime branch in one file (cheaper than the
  CUDA-per-arch or shim route), but it hasn't been done.

## 4. External validation — independent ORNL benchmark of Mojo kernel quality

Godoy, Melnichenko, Valero-Lara, … Vetter (Oak Ridge National Lab), *"Mojo:
MLIR-Based Performance-Portable HPC Science Kernels on GPUs,"* SC'25 Workshops
(arXiv:2509.21039). Independent (non-Modular) study, Mojo vs vendor CUDA/HIP on
**H100 and MI300A** (not GB10 — so our sm_121 numbers are complementary).

**Mojo efficiency vs vendor (Table 5), 1.0 = parity:**

| Kernel (class) | H100 vs CUDA | MI300A vs HIP |
|---|---|---|
| 7-point stencil (mem-bound) | 0.82–0.87 | 1.00 |
| BabelStream Copy/Mul/Add/Triad (mem-bound) | **1.01–1.02** | 1.00 |
| BabelStream Dot (shared-mem reduction) | 0.78 | 1.00 |
| miniBUDE (compute-bound) | 0.59–0.82 | 0.38 |
| Hartree-Fock (atomics) | up to **2.5× faster** than CUDA (small); 55× slower at a=1024 | ≪ HIP |

Key conclusions, and how they map to our GB10 data:

- **"Mojo's single-GPU performance is on par with CUDA/HIP for memory-bound
  kernels" with zero hand-tuning.** GB10 LLM inference is *memory-bandwidth-bound*
  — so the regime that matters on GB10 is the one where Mojo is already competitive.
  Our 1.94 tok/s for gemma-4-31B is a **bf16-footprint** problem (no low-bit path),
  not a kernel-quality problem — consistent with this.
- **Compute-bound gaps are mainly "lack of fast-math" + register pressure.** Mojo
  uses more registers/thread than CUDA (24 vs 21 stencil, 26 vs 20 BabelStream).
  Our 56 TFLOP/s generic `multistage_gemm` sits in this "runs its own code, not
  peak-tuned" zone.
- Mojo can **beat** CUDA (BabelStream, small atomics) and can fall off a cliff
  (Hartree-Fock a=1024) — workload-dependent, not a constant fraction.
- Mojo AOT binaries work out-of-the-box with **`ncu`/`rocprof`** — so our GB10
  kernels are profilable with Nsight Compute for deeper register/occupancy analysis.

## 5. Bottom line

On GB10 today, Mojo runs **its own correctly-sm_121a-compiled kernels** end-to-end
(no CUDA fallback), competitive-by-design for the bandwidth-bound regime that LLM
inference lives in, but lacking an arch-*specialized* path (matmul falls to generic
`multistage_gemm`; MoE to naive). Every framework is in the same "untuned on GB10"
boat — community CUDA doesn't target sm_121, ThunderKittens needed a hand-written
shim (3 kernels), Triton JITs but untuned. Mojo's edge is that closing the gap is a
single comptime branch in one multi-arch source rather than a new per-arch kernel.
