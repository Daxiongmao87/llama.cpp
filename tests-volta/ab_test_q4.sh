#!/bin/bash
# A/B: vanilla glimmer vs volta-patched llama-server, 4x22K-token prefill burst, qwen3.8-4b-empero
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

echo "=== A: VANILLA glimmer (port 8087) ==="
LD_LIBRARY_PATH=/mnt/hdd/ai/build/llama-cpp-glimmer/build/bin /mnt/hdd/ai/build/llama-cpp-glimmer/build/bin/llama-server $COMMON --port 8087 > /tmp/vanilla-8087-q4.log 2>&1 &
VPID=$!
wait_ready 8087 || { tail -30 /tmp/vanilla-8087-q4.log; kill $VPID; exit 1; }
python3 $BURST 8087 4 22000 qwen4b
kill $VPID; wait $VPID 2>/dev/null
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0

echo "=== B: VOLTA patched (port 8086) ==="
/home/agent/llama-cpp-volta/build/bin/llama-server $COMMON --port 8086 > /tmp/volta-8086-q4.log 2>&1 &
TPID=$!
wait_ready 8086 || { tail -30 /tmp/volta-8086-q4.log; kill $TPID; exit 1; }
python3 $BURST 8086 4 22000 qwen4b
kill $TPID; wait $TPID 2>/dev/null
echo "AB-DONE"
