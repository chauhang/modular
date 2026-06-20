#!/usr/bin/env bash
# Probe which MAX serving paths actually work on GB10 (sm_121).
# For each (model, encoding, device) it starts `max serve` in the container,
# polls the OpenAI endpoint until ready or the process dies, records PASS/FAIL
# + the error signature, then tears the container down. Sequential only.
#
# Usage: ./probe_max_paths.sh phaseA   (fast, tiny dense model, all encodings)
#        ./probe_max_paths.sh phaseB   (real Qwen3.6-35B-A3B MoE checkpoints)
set -u

IMAGE="${MAX_IMAGE:-docker.modular.com/modular/max-nvidia-full:latest}"
HFCACHE="$HOME/.cache/huggingface"
MODELS="$HOME/ard/models"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/results"
mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/path_probe_results.tsv"
PORT=8000
READY_TIMEOUT="${READY_TIMEOUT:-600}"   # seconds to wait for weight load + graph compile + server up

[ -f "$RESULTS" ] || printf "timestamp\tlabel\tmodel\tencoding\tdevice\tresult\tdetail\n" > "$RESULTS"

probe() {
  local label="$1" model="$2" enc="$3" dev="$4"; shift 4
  local extra=("$@")
  local name="maxprobe_$$"
  echo "=================================================================="
  echo "PROBE: $label  | model=$model enc=$enc dev=$dev"
  docker rm -f "$name" >/dev/null 2>&1

  docker run -d --name "$name" --gpus=all -p ${PORT}:8000 \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
    -v "$HFCACHE":/hf -v "$MODELS":/models \
    "$IMAGE" --model-path "$model" \
    --quantization-encoding "$enc" --devices "$dev" \
    --max-length 2048 --max-batch-size 1 "${extra[@]}" >/dev/null 2>&1

  local t=0 result="" detail=""
  while [ $t -lt $READY_TIMEOUT ]; do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      result="PASS"; detail="server_ready_${t}s"; break
    fi
    if ! docker ps -q --filter "name=$name" | grep -q .; then
      result="FAIL"
      detail="$(docker logs "$name" 2>&1 | grep -iE 'error|assert|not supported|unsupported|sm_|traceback|exception' | tail -3 | tr '\n' '|' | cut -c1-300)"
      break
    fi
    sleep 5; t=$((t+5))
  done
  [ -z "$result" ] && { result="TIMEOUT"; detail="$(docker logs "$name" 2>&1 | tail -3 | tr '\n' '|' | cut -c1-300)"; }

  # On PASS, capture a one-shot generation sanity check + device actually used
  if [ "$result" = "PASS" ]; then
    local gen; gen=$(curl -sf "http://localhost:${PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"prompt\":\"The capital of France is\",\"max_tokens\":8}" 2>/dev/null \
      | head -c 200 | tr '\n' ' ')
    detail="$detail; gen=${gen:0:120}"
  fi

  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$label" "$model" "$enc" "$dev" "$result" "$detail" >> "$RESULTS"
  echo "RESULT: $result  ($detail)"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 3
}

TINY="Qwen/Qwen2.5-0.5B-Instruct"

case "${1:-phaseA}" in
  phaseA)
    probe "tiny-bf16-gpu"  "$TINY" bfloat16        gpu
    probe "tiny-fp8-gpu"   "$TINY" float8_e4m3fn   gpu
    probe "tiny-fp4-gpu"   "$TINY" float4_e2m1fnx2 gpu
    probe "tiny-q4k-cpu"   "$TINY" q4_k            cpu
    ;;
  phaseB)
    # decisive-first: native quantized checkpoints test the real sm_121 kernel paths
    probe "moe-nvfp4-gpu"  "RedHatAI/Qwen3.6-35B-A3B-NVFP4" float4_e2m1fnx2 gpu   # 24GB
    probe "moe-fp8-gpu"    "Qwen/Qwen3.6-35B-A3B-FP8"      float8_e4m3fn   gpu    # 35GB
    probe "moe-q4k-gpu"    "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" q4_k   gpu    # expect reject (cpu-only)
    probe "moe-bf16-gpu"   "Qwen/Qwen3.6-35B-A3B"          bfloat16        gpu    # 70GB, confirms MoE bf16 path
    ;;
  *) echo "unknown phase: $1"; exit 1;;
esac

echo; echo "=== results so far ==="; column -t -s$'\t' "$RESULTS"
