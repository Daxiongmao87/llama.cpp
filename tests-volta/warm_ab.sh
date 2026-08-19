#!/bin/bash
# Warm A/B: cold burst (absorbs one-time costs) + warm burst (production-equivalent)
MODEL=/mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf
COMMON="--alias qwen4b --host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 131072 --cache-ram 6144 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --model $MODEL --n-predict 8192 --n-gpu-layers all --parallel 4 --split-mode none --timeout 28800 --ubatch-size 1024 --cache-idle-slots -v"
BURST=/home/agent/llama-cpp-volta/tests-volta/prefill_burst.py

wait_ready() {
  for i in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; then echo "READY after ${i}x2s"; return 0; fi
    sleep 2
  done
  echo "NOT_READY"; return 1
}

run_both() {
  local BIN=$1 LDV=$2 PORT=$3 TAG=$4
  echo "===== $TAG (port $PORT) ====="
  [ -n "$LDV" ] && export LD_LIBRARY_PATH=$LDV
  $BIN $COMMON --port $PORT > /tmp/warm-$TAG.log 2>&1 &
  local PID=$!
  wait_ready $PORT || { tail -20 /tmp/warm-$TAG.log; kill $PID; return 1; }
  echo "--- $TAG burst 1 (COLD) ---"
  python3 $BURST $PORT 4 22000 qwen4b
  echo "--- $TAG burst 2 (WARM) ---"
  python3 $BURST $PORT 4 22000 qwen4b W2
  kill $PID; wait $PID 2>/dev/null
  sleep 8
}

run_both /mnt/hdd/ai/build/llama-cpp-glimmer/build/bin/llama-server /mnt/hdd/ai/build/llama-cpp-glimmer/build/bin 8087 VANILLA
run_both /home/agent/llama-cpp-volta/build/bin/llama-server "" 8086 VOLTA
echo "WARM-DONE"
