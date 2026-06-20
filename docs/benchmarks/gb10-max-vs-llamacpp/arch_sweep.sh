#!/usr/bin/env bash
# Sweep local HF models to classify which ones THIS MAX image recognizes.
# Fast: the "architecture not available" rejection fires at config validation
# (~15s), so a short window distinguishes UNSUPPORTED vs arch-accepted.
set -u
IMAGE="docker.modular.com/modular/max-nvidia-full:latest"
HFCACHE="$HOME/.cache/huggingface"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/results"; mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/arch_sweep_results.tsv"
WINDOW="${WINDOW:-75}"   # seconds: enough to catch arch-reject and reach weight-load
[ -f "$RESULTS" ] || printf "timestamp\tmodel\tclass\tdetail\n" > "$RESULTS"

sweep() {
  local model="$1"; local name="maxsweep_$$"
  echo "============== $model =============="
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --gpus=all -p 8000:8000 \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf -v "$HFCACHE":/hf \
    "$IMAGE" --model-path "$model" --devices gpu --max-length 2048 --max-batch-size 1 >/dev/null 2>&1
  local t=0 class="" detail=""
  while [ $t -lt $WINDOW ]; do
    if curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; then class="SUPPORTED_SERVES"; detail="ready_${t}s"; break; fi
    local logs; logs="$(docker logs "$name" 2>&1)"
    if echo "$logs" | grep -qi "architecture not available"; then class="UNSUPPORTED_ARCH"; detail="arch_not_registered"; break; fi
    if echo "$logs" | grep -qiE "not supported by MAX engine"; then class="UNSUPPORTED_ENCODING"; detail="$(echo "$logs"|grep -i 'not supported by MAX'|tail -1|cut -c1-160)"; break; fi
    if ! docker ps -q --filter "name=$name" | grep -q .; then class="FAIL_OTHER"; detail="$(echo "$logs"|grep -iE 'error|exception|traceback'|tail -2|tr '\n' '|'|cut -c1-200)"; break; fi
    sleep 5; t=$((t+5))
  done
  # still running, no arch reject => arch accepted, was loading/compiling
  [ -z "$class" ] && { class="SUPPORTED_LOADING"; detail="arch_ok_loading_at_${WINDOW}s"; }
  printf "%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" "$class" "$detail" >> "$RESULTS"
  echo "  -> $class ($detail)"
  docker rm -f "$name" >/dev/null 2>&1; sleep 2
}

# Remaining local models (part 2). Already swept: Qwen2.5-0.5B, Qwen3-4B-Instruct,
# Qwen3.6-27B (supported), Qwen3.6-35B-A3B (unsupported MoE).
for m in \
  "meta-llama/Llama-3.2-1B-Instruct" \
  "Qwen/Qwen3-4B-Base" \
  "Qwen/Qwen3Guard-Gen-0.6B" \
  "google/gemma-2-2b-it" \
  "google/gemma-4-E2B-it" \
  "google/gemma-4-31B-it" \
  "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
  "z-lab/Qwen3.6-35B-A3B-DFlash" ; do
  sweep "$m"
done
echo; echo "=== arch sweep ==="; column -t -s$'\t' "$RESULTS"
