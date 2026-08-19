#!/usr/bin/env bash
# Cache-size A/B: same churn workload, NOUNI, vary --cache-ram (2GB baseline already measured)
set -uo pipefail
MODEL=/mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf
BIN=/home/agent/llama-cpp-volta/build/bin/llama-server
COMMON="--alias qwen4b --host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --model $MODEL --n-predict 8192 --n-gpu-layers all --parallel 4 --split-mode none --timeout 28800 --ubatch-size 1024 --cache-idle-slots"

for GB in 4 6; do
  echo "===== ARM: CACHE_${GB}GB ====="
  $BIN $COMMON --cache-ram $((GB * 1024)) --port 8086 > /tmp/churn-C${GB}G.log 2>&1 &
  PID=$!
  for i in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8086/v1/models >/dev/null 2>&1; then
      echo "READY after $((i*2))s"
      grep -E 'kv_unified|prompt cache is enabled' /tmp/churn-C${GB}G.log | head -2
      break
    fi
    if ! kill -0 $PID 2>/dev/null; then echo "SERVER DIED"; tail -30 /tmp/churn-C${GB}G.log; exit 1; fi
    sleep 2
  done
  python3 /home/agent/llama-cpp-volta/tests-volta/churn_test2.py 8086 8 qwen4b
  kill $PID; wait $PID 2>/dev/null
  sleep 8
done
echo CHURN-CACHE-AB-DONE
