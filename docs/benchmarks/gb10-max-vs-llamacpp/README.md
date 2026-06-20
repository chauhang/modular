# MAX Docker vs llama.cpp — GGUF serving benchmark on GB10

First end-to-end slice (see plan
`docs/plans/2026-06-20-001-feat-max-docker-vs-llamacpp-gguf-plan.md`).

## Hardware

- NVIDIA **GB10** (DGX Spark), compute capability **sm_121** (cc 12.1), aarch64
- 128 GB unified LPDDR5X (~273 GB/s, bandwidth-bound), CUDA 13.0, driver 580

## U1 — MAX Docker image (aarch64)

**Result: arm64 image exists.** The `:latest` tag is a multi-arch manifest
covering both linux/arm64 and linux/amd64, so container-first is viable on GB10.

| Field | Value |
|---|---|
| Image | `docker.modular.com/modular/max-nvidia-full:latest` |
| arm64 digest | `sha256:7ae201eabdf3852a5c0f7645193e27b31f053327c8e695fa4b16f12ea54a2457` |
| amd64 digest | `sha256:711e98e8a49b8303059884d7842bd212b9282dd02293f15d98188b1898e70012` |
| Host arch | aarch64 → arm64 manifest auto-selected on pull |

Pull:

```bash
docker pull docker.modular.com/modular/max-nvidia-full:latest
```

Run (GPU, mount HF cache + local GGUFs):

```bash
docker run --gpus=1 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/ard/models:/models \
  docker.modular.com/modular/max-nvidia-full:latest \
  serve --weight-path /models/<file>.gguf --devices gpu:0 --port 8000
```

Verify the GPU is visible from inside the container:

```bash
docker run --rm --gpus=1 docker.modular.com/modular/max-nvidia-full:latest \
  bash -lc 'nvidia-smi --query-gpu=name,compute_cap --format=csv'
# expect: NVIDIA GB10, 12.1
```

## Comparison setup

| Engine | Binary | Device for Q4_K_M | Endpoint |
|---|---|---|---|
| MAX | `max serve` (container) | **CPU** (q4_k is CPU-only in MAX) | `:8000/v1/chat/completions` |
| llama.cpp | `~/ard/llama.cpp/build/bin/llama-server` | **GPU** (`-ngl 99`) | `:8001/v1/chat/completions` |

Both driven by the same client: `max benchmark` (reports tok/s, TTFT, TPOT, ITL,
latency percentiles).

> **Note (honest framing):** this first slice is MAX-on-CPU vs llama.cpp-on-GPU
> for the Q4_K weights — it proves the pipeline and confirms the CPU-only path,
> but is **not** the GPU-vs-GPU verdict. The bf16 GPU comparison is the deferred
> next slice.

## Status

- [x] U1 — arm64 image confirmed; pull initiated
- [ ] U2 — small-model smoke (serve + `max benchmark`)
- [ ] U3 — Qwen3.6-35B-A3B Q4_K_M head-to-head
- [ ] U4 — capture + REPORT.md
