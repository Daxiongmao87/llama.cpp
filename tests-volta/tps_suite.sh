#!/usr/bin/env bash
# TPS suite: minicpm5-1b + qwen3.8-2b + ling-3.0-tiny (+ qwen4b baseline)
# Per model: short-ctx decode, 24K-ctx decode (subagent-realistic), prefill burst.
set -uo pipefail
BIN=/home/agent/llama-cpp-volta/build/bin/llama-server
T=/home/agent/llama-cpp-volta/tests-volta
COMMON="--host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 131072 --cache-ram 6144 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --slot-sim-use-cache --cont-batch-split --n-predict 8192 --n-gpu-layers all --parallel 4 --split-mode none --timeout 28800 --ubatch-size 1024 --cache-idle-slots"

run_model() {
  local ALIAS=$1 MODEL=$2
  echo "===== MODEL: $ALIAS ====="
  $BIN --alias $ALIAS $COMMON --model $MODEL --port 8086 > /tmp/tps-$ALIAS.log 2>&1 &
  local PID=$!
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:8086/v1/models >/dev/null 2>&1 && break
    if ! kill -0 $PID 2>/dev/null; then echo "SERVER DIED"; tail -20 /tmp/tps-$ALIAS.log; return 1; fi
    sleep 2
  done
  echo "--- decode tps (4 streams x 200 tok, ~1K ctx):"
  python3 $T/decode_tps.py 8086 4 200 1000 $ALIAS
  echo "--- decode tps (4 streams x 200 tok, 24K ctx = subagent-realistic):"
  python3 $T/decode_tps.py 8086 4 200 24000 $ALIAS
  echo "--- prefill burst (4 x ~22K tok TTFT):"
  python3 $T/prefill_burst.py 8086 4 22000 $ALIAS
  kill $PID; wait $PID 2>/dev/null
  sleep 8
}

run_model minicpm5 /mnt/hdd/ai/models/llm/minicpm5-1b/MiniCPM5-1B-Q4_K_M.gguf
run_model qwen2b /mnt/hdd/ai/models/llm/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf
run_model ling /mnt/hdd/ai/models/llm/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf
run_model qwen4b /mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf
echo TPS-SUITE-DONE
