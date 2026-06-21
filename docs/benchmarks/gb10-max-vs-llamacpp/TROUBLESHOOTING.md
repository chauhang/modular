# MAX on GB10 (DGX Spark, sm_121) — Troubleshooting Guide

Hard-won notes from debugging MAX serving + Mojo kernels on the NVIDIA GB10
(DGX Spark): **aarch64 Grace CPU + Blackwell sm_121, 128 GB unified LPDDR5X,
CUDA 13, driver 580**. The defining trait of this box is **unified memory** —
one 128 GB pool shared by CPU and GPU — which breaks several assumptions MAX
(and most GPU stacks) make about discrete GPUs.

> **The #1 lesson of the day:** when a model "won't run," **measure which
> process owns the memory before blaming the compiler.** We burned hours
> assuming the Mojo compiler was exploding. It wasn't. The real cause was MAX
> reserving ~90% of the *unified* pool as a device arena. See
> [The big one](#the-big-one-worker-terminated--oom-on-model-load).

---

## Quick reference

| Symptom | Most likely cause | First thing to try |
|---|---|---|
| Worker `Terminated` / earlyoom SIGTERM during "Initializing model pipeline", even for tiny models | **Device-memory arena reserves ~90% of unified pool** | `-e MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=15` (tune per model) |
| `Terminated` with no MAX error/traceback | It's an **OS/earlyoom kill** (below MAX's error handling), not a MAX crash | Trace memory + process table (below) |
| Exits ~15s with lots of free RAM, "don't leave room for KV cache" | Arena **too small** for weights+activations | Raise `MEMORY_MANAGER_SIZE_PERCENT` |
| Serves but **empty/garbage output** at very low free RAM | **KV cache starved** | Raise arena slightly / lower `--max-length` |
| `Architecture '…' not found in registry` for a custom arch | Registry resolves task **before** importing custom archs | Add `--task text_generation` |
| `quantization_encoding of 'float8_e4m3fn'/'float4…' not supported` | fp8/fp4 are **SM100-gated**, not sm_121 | Use `bfloat16` (only GPU path) |
| head_dim-256 model slow (gemma-4, Qwen3.6) | **Flash attention falls to naive** on sm_121 | Inherent gap (#6570); see [Kernel coverage](#kernel-coverage-gaps) |
| `nvidia-smi` shows `N/A` for memory | Expected — **unified memory** has no separate GPU pool to report | Use `free -g` for memory state |

---

## The big one: worker `Terminated` / OOM on model load

**Symptom.** ~40-60s into startup, during/after the log line
`Initializing model pipeline...`, the worker dies. Happens even for a **0.5B
model** on a **clean box with 113 GB free**.

The **literal error** in the MAX logs (this is all you get — note it says
*nothing* about memory):
```json
{"levelname": "ERROR", "message": "Worker exception, shutting down: Terminated", ...}
```
```
  File ".../max/serve/pipelines/model_worker.py", line 479, in start_model_worker
  File ".../max/serve/process_control.py", line 174, in run_subprocess
    raise SubprocessExit(exitcode)
max.serve.process_control.SubprocessExit: Terminated
```

The **proof it's memory** lives in a *separate* log — `journalctl -u earlyoom`:
```
earlyoom: low memory! at or below SIGTERM limits: mem 5.00%, swap 100.00%
earlyoom: sending SIGTERM to process NNNNN uid 0 "python": badness 984, VmRSS 3.6 GiB
```

> **Why this is a trap:** MAX reports only `SubprocessExit: Terminated` — no
> OOM, no allocation traceback — because the worker was **killed externally by
> earlyoom (SIGTERM)** before MAX could detect anything. If you only read MAX's
> output you'll chase the wrong thing (we wrongly blamed the compiler for hours).
> **Always cross-check `journalctl -u earlyoom` and `dmesg` when you see
> `Terminated`.** Note the killed process's `VmRSS` is small (~3 GiB) — the real
> memory hog is a *device/unified allocation* not attributed to any process.

**Root cause.** MAX queries "GPU free memory", sees the **entire ~120 GB unified
pool**, and reserves ~90% of it as a **device-context memory arena**. On an 80 GB
discrete card this is a sane default; on 128 GB unified it consumes the host's
RAM too → free RAM collapses to ~5% → **earlyoom kills the worker** (it's
configured `-m 5 --prefer python`, so it targets the MAX worker by design).

The giveaway during diagnosis: the ~107 GB was **not in any process RSS** (the
python worker was only ~3 GB, and there was **no `mojo` compiler process at
all**) → it's a device/CUDA allocation, not host/compiler memory.

**Fix.**
```bash
-e MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=<N>
```
This caps the arena to N% of the pool. **Working example (Qwen2.5-0.5B):**
```bash
docker run -d --gpus=all \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -e MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=15 \
  -v ~/.cache/huggingface:/hf -p 8000:8000 \
  docker.modular.com/modular/max-nvidia-full:nightly \
  --model Qwen/Qwen2.5-0.5B-Instruct --devices gpu:0 \
  --max-length 2048 --max-batch-size 1
# spike 108G -> 33G, READY 96s, correct output ✅
```

**Tuning the percent (it is per-model, not a constant):**
- The arena must hold **weights + activations + KV cache**.
- Too low → exits fast with "don't leave room for KV cache".
- Too high → weight-load doubling OOMs (see below).
- Rough guide: `percent ≈ (weights_GB + ~15 GB activations + KV) / 128`, then
  leave host headroom above earlyoom's 5%.

| Model | weights | known-good-ish percent |
|---|---|---|
| 0.5B | ~1 GB | 15 |
| 4B | ~8 GB | ~20 |
| 27B | ~54 GB | ~56-60 (knife-edge; KV-starved — see caveat) |
| 31B | ~58 GB | very tight / may not have a clean window |

**Caveat for big models (27B/31B).** This is a *second*, real constraint, not the
arena bug: **54-58 GB of weights on 128 GB unified is inherently tight**, and
**weight loading transiently doubles** (host staging + device copy). Even at the
"right" percent, free RAM hits ~1 GB and KV is starved → empty output. These
models likely need quantization (blocked, see below) or smaller variants.

---

## Diagnostic methodology (what actually localizes the problem)

Don't guess. Run these three together:

1. **Memory trajectory** — sample `free -g` every few seconds during startup to
   find *when* the spike happens:
   ```bash
   while ...; do free -g | awk '/Mem:/{print "free",$4"G"}'; sleep 3; done
   ```
2. **Correlate with MAX log phase** — `docker logs <c>` and see what line is
   printing at the spike (`Initializing model pipeline` = graph/device alloc;
   actual compile would be a separate phase). A spike that **recovers** when the
   worker dies is a *reservation*, not a compile.
3. **Process table** — `ps -eo rss,comm,args --sort=-rss | head` during the
   spike. **If no process owns the memory, it's a device/CUDA/unified
   allocation** (not the compiler, not host RSS). This single check would have
   saved hours.

`journalctl -u earlyoom --since "5 min ago" | grep SIGTERM` confirms *who*
earlyoom killed and the trigger level.

---

## What does NOT work (don't waste time on these)

All tested today; none addressed the device-arena OOM:

- **`--device-memory-utilization`** (0.05–0.9) — this is the **KV-cache budget**,
  a *different* allocation. It does **not** cap the device arena. The 108 GB
  spike persisted at every value.
- **Compiler knobs** — `MOJO_COMPILE_OPTS="-j 1"`, `MAX_EAGER_OP_PRECOMPILE=0`,
  `MAX_JOBS`, `MODULAR_MOJO_NUM_THREADS` — all target the *compiler*, which was
  never the bottleneck.
- **MLIR/LLVM flags** — `mojo build` **rejects** `-mllvm` and
  `--mlir-disable-threading` ("unrecognized argument"). Only `-O`, `-j`, `-g`,
  `--sanitize` are exposed. No way to tune MLIR passes externally.
- **Reboot** — the box had **2 days uptime, 113 GB free, no leak** (`free`,
  `/proc/meminfo` clean, `dmesg` no GPU errors). Reboot wouldn't have helped.
- **Newer nightly / LLVM bump** — `dev2026062006` → `dev2026062114` (with LLVM
  bumps): **identical** behavior. Not a compiler regression.
- **Disabling/relaxing earlyoom** — *Don't.* It exists to prevent GB10 OOM
  **hard-hangs that need physical reboots**. Killing the borderline worker is it
  doing its job; fix the over-allocation instead.
- **Compile cache mount** (`MODULAR_CACHE_DIR`) — warms `.mojo_cache`/`.mogg_cache`
  but can't bootstrap past a crash, and wasn't the issue.

---

## Docker settings (verified against official `container.mdx`)

The official MAX container uses **bare-minimum flags** — and we confirmed adding
more is unnecessary:
- **No `--shm-size` / `--ipc=host` needed.** `/dev/shm` is 64 MB by default and
  that's fine — MAX uses it only for small ZMQ request/response tensors. (Avoid
  `--ipc=host`: it exposes the host's RAM-backed `/dev/shm`, and
  `MODULAR_MAX_SHM_WATERMARK=0.9` could then grab tens of GB of the unified pool.)
- **No `--ulimit memlock`** — MAX handles pinned memory internally.
- **`--memory=<N>g`** is a *blunt alternative* to the arena env var (caps what
  MAX sees as available → smaller arena), but the env var is cleaner and tunable.
- **Mount the model cache** for reuse: `-v ~/.cache/max_cache:/opt/venv/share/max/.max_cache`.

---

## Unified-memory gotchas (GB10-specific)

- **`nvidia-smi` reports `N/A` for memory** — there's no separate GPU pool;
  use `free -g`. A "leaked GPU allocation" would show up as host RAM in `free`.
- **Weight-load doubling** — loading N GB of weights transiently uses ~2N
  (host staging + device copy). A 54 GB model briefly needs ~108 GB.
- **Discrete-GPU assumptions break** — "reserve 90% of GPU memory" is fine on
  80 GB HBM, fatal on 128 GB shared. Watch for any "% of GPU memory" default.
- **The compiler uses *host* RAM** — on a normal H100 server the compile lands in
  abundant host RAM (separate from the 80 GB GPU); on GB10 host RAM *is* the GPU
  pool, so compile + runtime + model all compete. (This is real but was **not**
  today's OOM cause — that was the arena.)

---

## Kernel coverage gaps (sm_121)

- **Attention head_dim 256 → naive kernel.** Flash attention requires
  `sm_90`/`sm_100` for head_dim 256 (shared-memory: sm_121 has ~100 KB vs
  228 KB on H100/B200; the 256-tiles don't fit). Measured cost via `bench_mha`:
  **flash@128 = 2324 GFLOP/s vs naive@256 = 559 GFLOP/s (~4× slower)**. Affects
  **gemma-4 and Qwen3.6** (both head_dim 256). Tracked under #6570.
- **Quantization is bf16-only on GPU:**
  - `float8_e4m3fn`, `float4_e2m1fnx2` → **SM100-gated** (B100/B200/B300), not sm_121.
  - `q4_k`, `q6_k` → **CPU-only**.
  - GPTQ → GPU-capable but **Llama-arch only**.
  - ⇒ big models can't be shrunk → stay memory-bound (see big-model caveat).

---

## Benchmarking Mojo kernels on GB10

- **Use the container's mojo, not the hermetic bazel one.** The bazel toolchain
  mojo can't build standalone (`unable to locate module 'std'`). The container
  mojo resolves all kernel imports and accepts `-D` defines:
  ```bash
  docker run --rm --gpus=all -v $(pwd):/repo \
    --entrypoint bash docker.modular.com/modular/max-nvidia-full:nightly -c '
    cd /repo && export KERNEL_BENCHMARKS_ROOT=/repo/max/kernels/benchmarks
    /opt/venv/bin/mojo build -D depth=256 \
      max/kernels/benchmarks/gpu/nn/bench_mha.mojo -o /tmp/b && /tmp/b'
  ```
- `kbench` needs Python deps (pandas/click/rich) and the hermetic mojo (which
  fails standalone) — not worth it here; the container-mojo recipe is simpler.
- The effective target on GB10: `--target-cpu cortex-x925`,
  `--target-accelerator nvidia:sm_121a` (`mojo build --print-effective-target`).

---

## Op logging (limited use)

`-e MODULAR_MAX_DEBUG_OP_LOG_LEVEL=TRACE` (value is **uppercase** — looked up by
enum name) emits `[OP] LAUNCH/COMPLETE` with op fusion + device targets. Caveats:
- It **destabilizes a full serve** (per-op tracing overhead → worker too slow →
  killed). Use only for a single forward pass via the Python API.
- It shows **fused graph ops**, not the comptime kernel variant — so it **cannot**
  confirm flash-vs-naive attention (use `bench_mha` for that).

---

## Environment / tooling notes

- **Always preflight** before a run: `bash ~/ard/sparky/gpu_preflight.sh <N>` —
  stops leftover servers, drops caches, kills lingering GPU apps, gates on free
  memory. It does **not** tune the device arena (that's the env var above).
- **One engine at a time** — the box serves a single model; don't run concurrent
  containers.
- **earlyoom config**: `-m 5 -s 100 --prefer python` — kills the MAX worker at
  5% free memory. Leave it on.
- **Watch for other GPU workloads** — e.g. a stray `megakernels/scripts/generate.py`
  was found competing for the unified pool during a test. Check `ps` for
  unexpected memory users.

---

## Models tested (status snapshot)

- ✅ **Qwen2.5-0.5B-Instruct** — serves cleanly with the arena fix, correct output.
- ⚠️ **Qwen3.6-27B** (dense) — served earlier at 4.59 tok/s; now knife-edge under
  the arena fix (KV-starved). head_dim 256 → naive attention.
- ⚠️ **gemma-4-31B-it** — served earlier at 1.94 tok/s (slow: naive attention +
  large-vocab sampling); now memory-tight.
- ❌ **Qwen3.6-35B-A3B** (`Qwen3_5MoeForConditionalGeneration`) — unsupported arch;
  custom bring-up compiles but the bf16 MoE is too large to load.
- ❌ fp8/fp4/q4_k variants — all unsupported on sm_121 GPU.

---

## Appendix: full flags & environment-variable reference

Everything we touched today, with what it does and whether it helped on GB10.
`✅` = useful, `❌` = tested, did **not** help our OOM, `ℹ️` = informational.

### Memory / device

| Flag / env var | Default | What it does | GB10 finding |
|---|---|---|---|
| `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` | ~unset (≈90% behavior) | Caps the **device-memory arena** as % of the (unified) pool | ✅ **THE fix.** `=15` for 0.5B; tune up per model |
| `--device-memory-utilization` | 0.9 | **KV-cache budget** = `free × util − weights` | ❌ does not cap the arena; 0.05–0.9 all spiked 108 GB |
| `MODULAR_MAX_SHM_WATERMARK` | 0.9 | % of `/dev/shm` for ZMQ shared memory | ℹ️ irrelevant (shm is 64 MB; don't enlarge) |
| `--kv-cache-page-size` / `--kv-cache-format` | model default | KV cache page sizing / dtype | ℹ️ page_size must be ≥ MHA tile (= head_dim) |
| `--memory=<N>g` (docker) | unlimited | Caps container RAM → MAX sees less → smaller arena | ℹ️ blunt alternative to the env var |
| `MODULAR_AUTO_CAST_WEIGHTS` | true | Auto-cast fp32↔bf16 on dtype mismatch | ℹ️ |

### Debugging / logging

| Flag / env var | What it does | Notes |
|---|---|---|
| `MODULAR_MAX_DEBUG_OP_LOG_LEVEL=TRACE` | Emits `[OP] LAUNCH/COMPLETE` op trace | value is **UPPERCASE** (enum name); destabilizes full serve; shows fused ops, not kernel variant |
| `MODULAR_DEBUG=<opts>` | Comma-list of debug options (below) | e.g. `MODULAR_DEBUG=stack-trace-on-crash,source-tracebacks` |
| ↳ `stack-trace-on-crash` / `stack-trace-on-error` | Mojo stack traces on crash/error | **won't fire on earlyoom SIGTERM** (external kill) |
| ↳ `source-tracebacks` | Python source locations in errors | |
| ↳ `nan-check` (+ `nan-check-stride`) | NaN/Inf checks on sampled kernel outputs | accuracy debugging |
| ↳ `assert-level=none\|warn\|safe\|all` | Mojo stdlib assertion level | |
| ↳ `ir-output-dir=<path>` | Dumps intermediate **compiler IR** | to inspect what's being compiled |
| ↳ `device-sync-mode` | Forces synchronous GPU execution | |
| ↳ `uninitialized-read-check` | Detects uninitialized-memory reads | |
| ↳ `print-style=compact\|full\|binary` | Tensor debug print format | |
| `MAX_SERVE_LOGS_CONSOLE_LEVEL=DEBUG` | Server console verbosity | `CRITICAL…DEBUG` |
| `MODULAR_STRUCTURED_LOGGING` | JSON structured logs | default `1` |
| `-D LOGGING_LEVEL=trace` (mojo build) | Compile-time op logging into a kernel binary | for `mojo build`, not serving |
| `./bazelw run --config=mojo-trace` | Op logging via bazel | |
| `session.set_mojo_log_level(LogLevel.TRACE)` (Py API) | Op logging for one InferenceSession | the **stable** way to op-log a single forward pass |

### Profiling

| Flag / env var | What it does | Notes |
|---|---|---|
| `--gpu-profiling [off\|on\|detailed]` | GPU profiling during serve | |
| `--profiling-enabled` / `--profiling-output-path` | Chrome-trace JSON output | `MODULAR_ENABLE_PROFILING=off\|on\|detailed` env equiv |
| `ncu --set full --import-source yes …` | Nsight Compute kernel profiling | build with `mojo build --debug-level=line-tables` first; works on GB10 AOT binaries |

### Compile / cache

| Flag / env var | What it does | GB10 finding |
|---|---|---|
| `MODULAR_CACHE_DIR=<path>` | Base dir for all caches (`.mojo_cache`, `.mogg_cache`) | ✅ mount to persist kernel cache across containers |
| `MODULAR_MAX_CACHE_DIR` | **Model-graph cache** (`$MODULAR_CACHE_DIR/.max_cache`) | the cache that skips the *whole* compile; only writes on a successful compile |
| `MODULAR_HOME` | Cache base when set (`$MODULAR_HOME/cache`) | default in-image `/opt/venv/share/max` (ephemeral) |
| `MAX_EAGER_OP_PRECOMPILE=0` | Lazy (on-first-use) vs eager op compile | ❌ didn't fix OOM (wasn't the compiler) |
| `MOJO_COMPILE_OPTS="-j N"` / `"-O0"` | Pass flags to the kernel JIT | ❌ no effect on the OOM |
| `MAX_JOBS` / `MODULAR_MOJO_NUM_THREADS` | Compile parallelism | ❌ wrong layer (not the compiler) |
| `MODULAR_NVPTX_COMPILER_PATH` | Use a system `ptxas` | untested fully; what #6396 tried |

### Mojo compiler (`mojo build`) — exposed flags only

| Flag | Notes |
|---|---|
| `-O / --optimization-level 0-3` | default 3 |
| `-j / --num-threads N` | default 0 = all threads |
| `-g / --debug-level` | `line-tables` for ncu |
| `--sanitize address\|thread` | |
| `--print-effective-target` | GB10: `cortex-x925`, `nvidia:sm_121a` |
| `--print-supported-targets` / `--print-supported-cpus` | |
| ~~`-mllvm`~~ / ~~`--mlir-*`~~ | **REJECTED** — no MLIR/LLVM pass-through exposed |

### Model bring-up / serve

| Flag / env var | What it does | Notes |
|---|---|---|
| `--task text_generation` | Forces the pipeline task | **required for custom `--custom-architectures`** (registry resolves task before importing them) |
| `--custom-architectures <port_dir>` | Load external arch module | passes `dirname` to `sys.path`, imports `basename` |
| `--no-device-graph-capture` | Disable CUDA graph capture | |
| `set_virtual_device_target_arch("sm_121")` (Py) | Compile **for** sm_121 on a CPU/virtual device | basis of off-box precompile (`precompile_pipeline.py`) |

### Quantization encodings on sm_121 GPU

| Encoding | sm_121 GPU? |
|---|---|
| `bfloat16` | ✅ only working GPU path |
| `float8_e4m3fn` (fp8) | ❌ SM100-gated |
| `float4_e2m1fnx2` (fp4 / NVFP4) | ❌ SM100-gated |
| `q4_k` / `q6_k` (GGUF) | ❌ CPU-only |
| GPTQ | ⚠️ GPU-capable but Llama-arch only |

---

*Last updated: 2026-06-21. Image: `max-nvidia-full:nightly`
(`Mojo 1.0.0b3.dev2026062114` / MAX `26.5.0.dev2026062114`).*
