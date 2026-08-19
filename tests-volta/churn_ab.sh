#!/bin/bash
# Churn A/B: --no-kv-unified vs --kv-unified, both on the volta build, tight RAM cache
MODEL=/mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf
COMMON="--alias qwen4b --host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 131072 --cache-ram 2048 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --model $MODEL --n-predict 8192 --n-gpu-layers all --parallel 4 --split-mode none --timeout 28800 --ubatch-size 1024 --cache-idle-slots -v"
unset LD_LIBRARY_PATH

run_arm() {
  local TAG=$1 KVFLAG=$2
  local ARGS="$COMMON"
  [ "$KVFLAG" = "unified" ] && ARGS=${COMMON/--no-kv-unified/ --kv-unified}
  echo "===== ARM: $TAG ====="
  /home/agent/llama-cpp-volta/build/bin/llama-server $ARGS --port 8086 > /tmp/churn-$TAG.log 2>&1 &
  local PID=$!
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:8086/v1/models >/dev/null 2>&1 && { echo "READY after ${i}x2s"; break; }
    sleep 2
  done
  grep -m1 'kv_unified' /tmp/churn-$TAG.log
  python3 /home/agent/llama-cpp-volta/tests-volta/churn_test2.py 8086 8 qwen4b
  kill $PID; wait $PID 2>/dev/null
  sleep 8
}

run_arm NOUNI nonunified
run_arm UNI unified
echo "CHURN-AB-DONE"
