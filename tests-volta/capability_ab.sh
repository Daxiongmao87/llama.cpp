#!/usr/bin/env bash
# Capability A/B: the same 4 models as tps_suite.sh, on one fixed build task.
# Sequential, one model on the GPU at a time, identical server flags and seeds.
# Companion to tps_suite.sh -- that one measures speed, this one measures output.
set -uo pipefail
BIN=/home/agent/llama-cpp-volta/build/bin/llama-server
T=/home/agent/llama-cpp-volta/tests-volta
OUT=${OUT:-/tmp/capability-web}
RUNS=${RUNS:-3}
PORT=8087
# --parallel 1: single-stream, so capability is not confounded by slot contention
COMMON="--host 127.0.0.1 --jinja --batch-size 1024 --ctx-size 32768 --cache-ram 2048 --cache-type-k q4_0 --cache-type-v q4_0 --device CUDA0 --flash-attn on --fit on --no-kv-unified --n-predict 8192 --n-gpu-layers all --parallel 1 --split-mode none --timeout 3600"

mkdir -p "$OUT"
# ONLY="lfm25 qwen4b" runs a subset and appends, so one model can be added to an
# existing result set without re-measuring (and re-truncating) the rest.
ONLY=${ONLY:-}
[ -z "$ONLY" ] && : > "$OUT/scores.txt"

run_model() {
  local ALIAS=$1 MODEL=$2
  if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qx "$ALIAS"; then return 0; fi
  echo "===== MODEL: $ALIAS ====="
  $BIN --alias "$ALIAS" $COMMON --model "$MODEL" --port $PORT > "/tmp/cap-$ALIAS.log" 2>&1 &
  local PID=$!
  for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break
    if ! kill -0 $PID 2>/dev/null; then echo "SERVER DIED"; tail -20 "/tmp/cap-$ALIAS.log"; return 1; fi
    sleep 2
  done
  echo "# $ALIAS <- $MODEL" >> "$OUT/scores.txt"
  # tee: the per-run check breakdown is the diagnostic, don't leave it in scrollback
  python3 "$T/capability_web.py" $PORT "$ALIAS" "$OUT" "$RUNS" 2>&1 | tee -a "$OUT/run.log"
  kill $PID; wait $PID 2>/dev/null
  sleep 8
}

# tps_suite.sh order, then the LFM2.5 arms (not in that suite) in size order.
# qwen4b is the empero finetune (dir *-empero), not stock.
#
# The three LFM arms are deliberate: lfm25 (Q6_K, the file we had) vs lfm25qad
# (QAD-Q4_0, same model) isolates Quantization-Aware Distillation -- identical
# architecture, only the quantization path differs. lfm12 is the new small one.
# Everything else here is Q4_K_M, so lfm25's Q6_K is the one quant outlier.
run_model minicpm5 /mnt/hdd/ai/models/llm/minicpm5-1b/MiniCPM5-1B-Q4_K_M.gguf
run_model qwen2b   /mnt/hdd/ai/models/llm/qwen3.8-2b-empero/Qwen3.8-2B-Q4_K_M.gguf
run_model lfm12    /mnt/hdd/ai/models/llm/lfm2.5-1.2b/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf
run_model lfm25    /mnt/hdd/ai/models/llm/lfm2.5-2.6b/LFM2.5-2.6B-Q6_K.gguf
run_model lfm25qad /mnt/hdd/ai/models/llm/lfm2.5-2.6b/LFM2.5-2.6B-QAD-Q4_0.gguf
run_model ling     /mnt/hdd/ai/models/llm/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf
run_model qwen4b   /mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q4_K_M.gguf

# --- quantization sweep: Q8_0 counterparts of the Q4 arms above ---
# Tests the hypothesis that quantization error costs more at low parameter
# counts: the Q4->Q8 gain should shrink as 1B -> 2B -> 4B.
# Each pair below is same-repo, same-vintage, so the delta is precision alone.
run_model minicpm5q8 /mnt/hdd/ai/models/llm/minicpm5-1b/MiniCPM5-1B-Q8_0.gguf
run_model qwen2bq8   /mnt/hdd/ai/models/llm/qwen3.8-2b-empero/Qwen3.8-2B-Q8_0.gguf
run_model qwen4bq8   /mnt/hdd/ai/models/llm/qwen3.8-4b-empero/Qwen3.8-4B-Q8_0.gguf

# --- LFM2.5 precision curve, all three from today's release ---
# lfm25q6new is the matched control for lfm25qad; the older lfm25 (Aug 8 build)
# is a different quantization vintage and must NOT be used as that control.
run_model lfm25q6new /mnt/hdd/ai/models/llm/lfm2.5-2.6b/LFM2.5-2.6B-Q6_K-20260819.gguf
run_model lfm25q8    /mnt/hdd/ai/models/llm/lfm2.5-2.6b/LFM2.5-2.6B-Q8_0.gguf

# --- ling: both sides quantized locally from the same bf16 source ---
# The pre-existing Ling Q4_K_M matches no public repo, so it cannot be paired
# with a downloaded Q8. These two are made from inclusionAI bf16 in one pass.
run_model lingq4loc  /mnt/hdd/ai/models/llm/ling-3.0-tiny-bf16/Ling-3.0-tiny-local-Q4_K_M.gguf
run_model lingq8loc  /mnt/hdd/ai/models/llm/ling-3.0-tiny-bf16/Ling-3.0-tiny-local-Q8_0.gguf
run_model lingq8bart /mnt/hdd/ai/models/llm/ling-3.0-tiny/Ling-3.0-tiny-Q8_0.gguf

echo
echo "===== SCORECARD ====="
cat "$OUT/scores.txt"
echo
echo "Artifacts in $OUT/ -- open the .html files to judge what the score cannot."
echo CAPABILITY-AB-DONE
