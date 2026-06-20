#!/usr/bin/env bash
# run_one.sh <name> <timeout> -- <serve args...>
set -u
name="$1"; TO="$2"; shift 3
docker rm -f "$name" >/dev/null 2>&1
docker run -d --name "$name" --gpus=all -p 8000:8000 -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -v "$HOME/.cache/huggingface":/hf -v "$HOME/ard/models":/models \
  docker.modular.com/modular/max-nvidia-full:nightly "$@" >/dev/null 2>&1
t=0
while [ $t -lt "$TO" ]; do
  if curl -sf localhost:8000/v1/models >/dev/null 2>&1; then
    echo "=== PASS (ready ${t}s) ==="
    curl -sf localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"m","prompt":"The capital of France is","max_tokens":12}' 2>/dev/null | head -c 300; echo
    break
  fi
  docker ps -q --filter "name=$name" | grep -q . || { echo "=== EXITED ==="; break; }
  sleep 10; t=$((t+10))
done
[ $t -ge "$TO" ] && echo "=== TIMEOUT ${TO}s ==="
echo "----- last logs -----"
docker logs "$name" 2>&1 | grep -iE "error|assert|runtime|not found|not supported|memory|gib|register|ready|started|listening|exceed" | tail -15
docker rm -f "$name" >/dev/null 2>&1
