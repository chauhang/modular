#!/usr/bin/env bash
# Start a MAX server for one model, wait until ready, run the streaming client,
# capture a results JSON with metadata, tear down. Single-stream (batch 1).
set -u
IMAGE="${MAX_IMAGE:-docker.modular.com/modular/max-nvidia-full:nightly}"
MODEL="${1:-google/gemma-4-31B-it}"
LABEL="${2:-gemma-4-31B-bf16}"
DIR="$(cd "$(dirname "$0")" && pwd)"; OUT="$DIR/results"; mkdir -p "$OUT"
name="maxbench"
docker rm -f "$name" >/dev/null 2>&1

echo "[$(date -u +%T)] starting MAX server for $MODEL ..."
docker run -d --name "$name" --gpus=all -p 8000:8000 -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface":/hf -v "$HOME/ard/models":/models \
  "$IMAGE" --model "$MODEL" --devices gpu:0 \
  --max-length 2048 --max-batch-size 1 --device-memory-utilization 0.95 >/dev/null 2>&1

t=0; ready=0
while [ $t -lt 900 ]; do
  curl -sf localhost:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
  docker ps -q --filter "name=$name" | grep -q . || { echo "server exited early:"; docker logs "$name" 2>&1 | tail -8; docker rm -f "$name" >/dev/null 2>&1; exit 1; }
  sleep 10; t=$((t+10))
done
[ "$ready" = 1 ] || { echo "TIMEOUT waiting for server"; docker rm -f "$name" >/dev/null 2>&1; exit 1; }
echo "[$(date -u +%T)] server ready in ${t}s; benchmarking ..."

RESULT="$OUT/serving_${LABEL}.json"
META=$(python3 -c "import json,subprocess,datetime;print(json.dumps({
 'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
 'image': '$IMAGE', 'model': '$MODEL', 'label': '$LABEL',
 'gpu': 'NVIDIA GB10 sm_121', 'device': 'gpu:0', 'precision': 'bfloat16',
 'max_length': 2048, 'batch_size': 1, 'ready_seconds': $t}))")
BENCH=$(python3 "$DIR/bench_client.py" "http://localhost:8000" "$MODEL")
python3 -c "import json,sys;m=json.loads('''$META''');b=json.loads('''$BENCH''');m['benchmark']=b;open('$RESULT','w').write(json.dumps(m,indent=2));print(json.dumps({k:m.get(k) or b.get(k) for k in ['model','ready_seconds']},indent=2));print('TTFT_mean_s:',b.get('ttft_s_mean'),' GEN_tok_s_mean:',b.get('gen_tok_s_mean'),' all:',b.get('gen_tok_s_all'))"
echo "saved -> $RESULT"
docker rm -f "$name" >/dev/null 2>&1
