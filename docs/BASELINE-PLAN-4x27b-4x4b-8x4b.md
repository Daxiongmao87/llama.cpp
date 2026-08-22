# Baseline Plan: 4x27B@128K / 4x4B@128K / 8x4B@32K
For 5-10 stream multi-agent engineering ecosystem. Investigation, not execution.

## 1. Hardware and current roster

- 2x V100 32GB PCIe PHB (no NVLink), CUDA0+CUDA1 usable, A2000 forbidden. Layer split only; row/tensor split is PCIe allreduce per layer.
- Current Volta build `Daxiongmao87/llama.cpp 697c96a` (`--slot-sim-use-cache`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_DISABLE_GRAPHS` absent per override). Router `c=786432` total.
- Files: `Qwen3.8-27B-IQ4_XS.gguf` 15G + `mmproj` 0.9G + `mtp-Q4_0` 1.6G, `Qwen3.8-4B-Q4_K_M.gguf` 2.6G (`Q8_0` 4.3G), `Qwen3.8-2B` 1.3G.

## 2. VRAM feasibility (receipts vs estimate)

**27B reference (router notes):** `fixed 14988 + mmproj 885 =15873`, `KV/q4_0 per 96K slot 3352 (incl 1090 MTP draft)`. Linear: `1K -> 34.9 MiB`.

- `4x96K` single: `15873+4*3352=29281` (3.5GB spare on 32GB) -> last known good.
- `5x96K` single: `32633` (135 spare, not viable).
- `4x128K` single: `15873+4*4469=33750` (>32768, not viable single).
- `4x128K` **layer split** across 2: total `33750`, `perGPU ~16875` (measured `6x128K` layer `21489/25017=46506` total, `perGPU 21-25GB`). **Fits with ~10-15GB headroom per GPU.**
- `6x131K` current `c=786432/6`: total `~42689`, perGPU `~21344` fits but leaves only `1.1GB` on CUDA0 after `lfm` (fragile, needs eviction slack).

**4B estimates (needs 1 real load to confirm, Q4_K_M 2.6G):** `fixed ~3200`, `KV/q4_0 per 128K ~900-1100` (scaled from 27B by `n_layers*n_kv_heads`), per 32K ~250.

- `4x4B@128K` single: `3200+4*1000=7200` (25GB spare) -> fits single V100, layer split not needed.
- `8x4B@32K` single: `3200+8*250=5200` (27GB spare) -> fits single, can run `16` if needed.
- `NVFP4` weight saving on 27B: `15G IQ4_XS -> ~14.5G NVFP4` (~0.5-1GB), on 4B `2.6->2.2` (~0.4GB). **Not capacity enabler**; the win is decode BW (`docs/NVFP4-VOLTA-PLAN.md:1` `QPN2 77%/71%`, `QPN8 82%`).

**Ecosystem packing:** 4x27B@128K layer needs both GPUs. It cannot co-reside with `4x4B@128K` or `8x4B@32K` simultaneously on `64GB` total unless time-shared. Practical ecosystem is `time-shared` or `3-GPU` (add A2000 if allowed) or `single-model-at-a-time` benchmarks. For “bare minimum concurrent” assume `4x27B` owns the box, then `4x4B` owns one GPU, `8x4B` owns one.

## 3. What “responsive” means to measure

- **Cold prefill TTFT** (fresh 128K, `--cache-ram 0`): `lucebox` 3090 `32K 1,168 tok/s`, `128K 774` warm `169s`, `248s` cold. Expect V100 `~500-700 tok/s` layer PHB `ubatch 512` (your `55K 520 tok/s 122s` log), `128K ~240-340s` cold. `Caleb` 3090 `121K` `TTFT 141s` baseline `MTP n=3 252s` cold, `15s->10s` with `cache-ram 8192` (cache hit).
- **Cached follow-up TTFT** (agent loop, `slot-sim-use-cache` on, `f_keep 1.0`): `~1-2s` for `128K` (Caleb cached `128K 1.91s` baseline `2.58s` MTP). This is the user-visible number between tool calls when pinned (`server-context.cpp:1527` LCP `0.99` -> no `prompt_save/load`).
- **Decode `tok/s/slot` concurrent:** `tps_suite.sh` prior: `27B 1 stream 36.4`, `2->20.6 (41.2 agg)`, `4->11.2 (44.7 agg)` (1.23x). With `MTP` `36->75` (+107% `sudoingX`) but cold prefill slower. `4x` vs `8x` decode `agg` flat, per-slot halved.
- **Churn hit%:** `churn_test2.py` `cached_tokens/prompt_tokens` `75%` with `slot-sim-use-cache` vs `65%` without (prevents `0-hit` full re-prefill each handoff).

NVFP4 helps only decode `agg` (+~20-30% expected on Volta, parity with 5090 was decode-only; prefill `4x` behind stays).

## 4. Prepared test matrix (no execution yet)

Use dedicated `llama-server` per baseline (restart, cold), `GGML_CUDA_GRAPHS=ON`, `flash-attn on`, `q4_0 KV` baseline, `MTP` built-in `spec-draft-n-max 4 p-min 0.60` vs `off`.

**A. 4x27B@128K layer** (owns both GPUs)
- Launch: `--model Qwen3.8-27B-IQ4_XS.gguf --mmproj mmproj... --device CUDA0,CUDA1 --split-mode layer --ctx-size 524288 --parallel 4 --batch-size 1024 --ubatch-size 1024 --cache-ram 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --cont-batch-split false --slot-sim-use-cache`
- Variants: `ubatch 512 vs 1024 vs 2048`, `MTP off vs n=4`, `KV q4_0 vs q8_0`, `+NVFP4 --volta-qpn` (when built, `GGML_CUDA_VOLTA_QPN=1`).
- Harness: `prefill_burst.py 8086 4 131072 qwen/qwen3.8-27b` (TTFT), `decode_tps.py 8086 4 200 131072`, `churn_test2.py 8086 8` + `churn_parse.py`.

**B. 4x4B@128K single** (fits one GPU, leave other free)
- Launch: `--model Qwen3.8-4B-Q4_K_M.gguf --device CUDA0 --ctx-size 524288 --parallel 4 --ubatch-size 1024` (same sweeps, `MTP` if 4B has head else off).
- Harness same, target `TTFT <30s` cold `128K` expected `~40-60s` on V100 single.

**C. 8x4B@32K single** (throughput swarm)
- Launch: `--device CUDA0 --ctx-size 262144 --parallel 8 --batch-size 1024 --ubatch-size 1024 --cont-batch-split true` (the `many-slot` case `cont-batch-split` helps per ling note).
- Harness: `decode_tps.py 8 200 32768`, `prefill_burst` `8x32K`.

**Instrumentation to capture per run:** `journalctl` `prompt processing .../ tok/s` + `slot print_timing`, `n_ctx` check, `VRAM` `nvidia-smi --query-gpu=memory.used`, `cached_tokens` hit%.

## 5. Why not run now

- 27B + 4B + 8x4B simultaneous exceeds `64GB` even layer split; needs serial launches with `systemctl stop` + `CUDA_VISIBLE_DEVICES` isolation.
- NVFP4 path not yet built (`mmq-config-volta.cuh` + `quantize_mmq_fp4` Volta gate) - running now would benchmark `Q4` only, not the `+20% decode` you hope from NVFP4. Plan keeps NVFP4 as one variant so tests are not wasted.
- `ubatch`/`batch` tuning for PHB needs cold `llama-bench pp131072 --no-warmup` baseline first to set `tok/s` ceiling before concurrent.

Next step when you say go: I will run `A` cold `TTFT` `ubatch` sweep single, then `A` concurrent `decode_tps`, then `B`/`C` serial, each with `5` runs median as `BAEM1n llm-bench` methodology.
