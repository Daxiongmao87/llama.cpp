#!/bin/bash
# VOLTA-only warm test (no glimmer LD_LIBRARY_PATH in scope)
MODEL=/mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf
COMMON="--alias qwen4b --host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 131072 --cache-ram 6144 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --model $MODEL --n-predict 8192 --n-gpu-layers all --parallel 4 --split-mode none --timeout 28800 --ubatch-size 1024 --cache-idle-slots -v"
BURST=/home/agent/llama-cpp-volta/tests-volta/prefill_burst.py
DEC=/home/agent/llama-cpp-volta/tests-volta/decode_tps.py

unset LD_LIBRARY_PATH

/home/agent/llama-cpp-volta/build/bin/llama-server $COMMON --port 8086 > /tmp/volta-warm.log 2>&1 &
PID=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8086/v1/models >/dev/null 2>&1 && { echo "READY after ${i}x2s"; break; }
  sleep 2
done

echo "--- VOLTA burst 1 (COLD, absorbs one-time warmup) ---"
python3 $BURST 8086 4 22000 qwen4b
echo "--- VOLTA burst 2 (WARM prefill) ---"
python3 $BURST 8086 4 22000 qwen4b W2
echo "--- VOLTA decode tps (4 streams x 200 tok) ---"
python3 $DEC 8086 4 200 1000 qwen4b
kill $PID; wait $PID 2>/dev/null
echo "VOLTA-WARM-DONE"
