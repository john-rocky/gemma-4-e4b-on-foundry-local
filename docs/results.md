# Results

[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) is not in the Foundry Local catalog ([microsoft/Foundry-Local#1232](https://github.com/microsoft/Foundry-Local/issues/1232) asks for it; the catalog has Gemma 4 E2B). This repo builds ONNX Runtime GenAI packages of E4B with [Mobius](https://github.com/onnxruntime/mobius), the exporter that the [olive-recipes Gemma 4 E2B recipes](https://github.com/microsoft/olive-recipes/tree/main/google-gemma-4-E2B-it) use, and runs them as local models in Foundry Local 0.13.0 on one Windows laptop.

The checks here are text in, text out. The packages also carry the vision and audio encoders; they load, but no image or audio prompt was checked.

Machine: MSI Stealth 16 AI+, Windows 11 build 26200.6899 (26200.9711 after an update on 11-06, see "After the Windows update"), Core Ultra 9 386H, 31.5 GB RAM, RTX 5080 Laptop 16 GB (driver 591.74), Intel Graphics (driver 32.0.101.8724), Intel AI Boost NPU (driver 32.0.100.4841). Gemma 4 E4B at revision `4ecc2ba63b`.

## In short

The numbers in this section and in the results table were measured on 11-06 before the laptop installed a Windows update at 17:31, except ONNX Runtime GenAI's int4 decode on CUDA without CUDA graphs (58.4 tokens/s), measured again on 11-09 (see Results). After the update, every form gives the same answers word for word, and most forms decode more slowly (see "After the Windows update").

- RTX GPU with 16 GB: the int4 package with an 8K-token context (steps 2 to 6) as a Foundry Local model decodes at 58.8 tokens/s, with 246 ms to the first token of a 933-token prompt. The daemon holds 13,294 MiB of GPU memory at most.
- Keep the context short on a 16 GB GPU. The export sets 131072 tokens. With that context, before the update, the int4 package filled the GPU over the 24 prompts, then took 15 s to the first token and decoded at 43.3 tokens/s. The fp16 package and the catalog E2B's CUDA form (the `mixed` int4 / int8 decoder) each ran out of GPU memory on 18 of the 24 prompts in a run (fp16 passed in another run, see "What to know"). After the update, the `mixed` package answered all 24 and peaked at 15,971 MiB.
- Less GPU memory: the int4 decoder with an 8K context and the fp16 embedding session on the CPU (`e4b-cuda-int4-8k-embcpu`, step 7). The daemon holds 4,042 MiB after load and 9,186 MiB at most, and decodes at 47.0 tokens/s.
- CPU only: the CPU package (step 8) on Foundry Local's CPU EP decodes at 11.6 tokens/s.
- Intel GPU and NPU: Foundry Local 0.13.0 does not run these packages there. Its OpenVINO EP fails to load them, and local models do not take the WebGPU EP. Outside Foundry Local, ONNX Runtime GenAI 0.19.0 decodes at 18.9 tokens/s with WebGPU on the Intel GPU and 7.9 with OpenVINO (step 11). The NPU does not compile the graph.
- CUDA graphs make decode 1.8 times as fast in ONNX Runtime GenAI (58.4 to 103.8 tokens/s for int4), but not in Foundry Local 0.13.0.

## Results

One laptop, 2026-11-06, before the Windows update (build 26200.6899). The 24 prompts in `fixture/` run greedy with 48 new tokens, against Hugging Face transformers 5.22.1 in bf16 on the same machine (`results/hf_ref.json`). The answer keys were written before any ONNX output was measured. A form passes when it answers right wherever the reference does (20 of the 24 prompts), with at most one miss. "Text = HF" counts answers identical to the reference's text. "First token = HF" counts the same first token, with the largest logit difference (GenAI only: Foundry Local's REST API returns text).

Speed: Foundry Local = a streamed chat completion over REST with a 933-token prompt and 160 new tokens; GenAI 0.19.0 = a 512-token prompt and 128 new tokens. Each is the median of 3 runs after a warm-up; the two runners used different prompts, so compare within one runner. Each Foundry Local form ran in a fresh daemon: load, the 24 prompts, then the speed run. The CPU run restarted the daemon and loaded the model again before its speed run. VRAM: for Foundry Local, the daemon's own dedicated GPU memory at its highest over that run (Windows performance counter); for GenAI, nvidia-smi's total.

The GenAI int4 decode on CUDA was measured again on 11-09 (build 26200.9711), with and without CUDA graphs, each in two processes that opt out of Windows power throttling at their start, before the script and its imports run (High QoS: `SetProcessInformation` with `ProcessPowerThrottling`), and in two that do not. With High QoS it decodes at 58.4 tokens/s without CUDA graphs, which replaces 54.7, and at 103.2 with them (103.8 kept, within 5 %). Without that call, started from an ssh session while the desktop is locked, the same run decodes at 36.9 tokens/s without CUDA graphs (`results/speed_high_qos_1109.txt`).

<!-- table: python make_table.py --public -->
| Form | Runner | EP / device | Keys (bar) | Text = HF | First token = HF | Prefill | Decode | VRAM peak |
|---|---|---|---|---|---|---|---|---|
| fp16, embedding session on CPU, 131072 context | Foundry Local | CUDA / RTX 5080 Laptop | not judged: 18 of 24 requests failed: `CUDA error: out of memory` | — | — | — | — | — |
| fp16, embedding session on CPU, 32K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 22/24 | — | TTFT 362 ms (933 tokens) | 28.6 tok/s | 15,966 MiB |
| int4 (RTN), 131072 context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 14,972 ms (933 tokens) | 43.3 tok/s | 15,997 MiB |
| int4 (RTN), 32K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 254 ms (933 tokens) | 53.6 tok/s | 15,920 MiB |
| int4 (RTN), 8K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 246 ms (933 tokens) | 58.8 tok/s | 13,294 MiB |
| int4 (RTN), 131072 context, CUDA graphs | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 15,327 ms (933 tokens) | 23.5 tok/s | 15,998 MiB |
| int4 (mixed, catalog E2B recipe), 131072 context | Foundry Local | CUDA / RTX 5080 Laptop | not judged: 18 of 24 requests failed: `CUDA error: out of memory` | — | — | — | — | — |
| int4 (mixed, catalog E2B recipe), 32K context | Foundry Local | CUDA / RTX 5080 Laptop | 19/20 PASS | 13/24 | — | TTFT 1,816 ms (933 tokens) | 49.3 tok/s | 15,986 MiB |
| int4 (mixed, catalog E2B recipe), 32K context, CUDA graphs | Foundry Local | CUDA / RTX 5080 Laptop | 19/20 PASS | 13/24 | — | TTFT 307 ms (933 tokens) | 49.7 tok/s | 15,939 MiB |
| int4 (RTN), int8 embedding on CPU, 8K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 421 ms (933 tokens) | 13.5 tok/s | 9,182 MiB |
| int4 (RTN), fp16 embedding on CPU, 8K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 362 ms (933 tokens) | 47.0 tok/s | 9,186 MiB |
| int4 (RTN), fp16 embedding on CPU, 8K context, CUDA arena by request | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 351 ms (933 tokens) | 48.4 tok/s | 9,186 MiB |
| int4 (RTN), fp16 embedding on CPU, 8K context, CUDA graphs | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 361 ms (933 tokens) | 47.8 tok/s | 8,192 MiB |
| fp16, embedding session on CPU | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 22/24 | 24/24 (max abs Δ 1.1) | — | — | — |
| int4 (RTN) | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.4) | 134 ms (512 tokens) | 58.4 tok/s | 11,352 MiB |
| int4 (RTN), CUDA graphs | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.4) | 127 ms (512 tokens) | 103.8 tok/s | 10,368 MiB |
| int4 (mixed, catalog E2B recipe) | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 19/20 PASS | 13/24 | 23/24 (max abs Δ 16.3) | 138 ms (512 tokens) | 50.7 tok/s | 13,136 MiB |
| int4 (mixed, catalog E2B recipe), CUDA graphs | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 19/20 PASS | 13/24 | 23/24 (max abs Δ 16.3) | 140 ms (512 tokens) | 70.5 tok/s | 13,792 MiB |
| int4 (RTN), int8 embedding on CPU, 8K context | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.4) | 235 ms (512 tokens) | 14.3 tok/s | 5,910 MiB |
| int4 (RTN) | GenAI 0.19.0 | WebGPU / RTX 5080 Laptop | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.5) | 1,748 ms (512 tokens) | 57.1 tok/s | 7,879 MiB |
| int4 (RTN) | GenAI 0.19.0 | WebGPU / Intel GPU | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.7) | 1,803 ms (512 tokens) | 18.9 tok/s | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / Intel GPU | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.2) | 2,178 ms (512 tokens) | 7.9 tok/s | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / CPU | 20/20 PASS | 13/24 | 22/24 (max abs Δ 33.9) | 5,785 ms (512 tokens) | 5.9 tok/s | — |
| int4 (RTN, CPU recipe) | Foundry Local | CPU / Core Ultra 9 386H | 20/20 PASS | 13/24 | — | TTFT 20,195 ms (933 tokens) | 11.6 tok/s | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | CPU / Core Ultra 9 386H | 20/20 PASS | 13/24 | 22/24 (max abs Δ 33.8) | 9,388 ms (512 tokens) | 15.6 tok/s | — |
| int4 (RTN) | Foundry Local | WebGPU / Intel GPU | load fails: `unknown execution provider for local model` (registered with `--execution-provider WebGpuExecutionProvider`) | — | — | — | — | — |
| int4 (RTN, CPU recipe) | Foundry Local | OpenVINO / Intel GPU, NPU, CPU | load fails on all three devices: `OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build` (Foundry-Local#1261) | — | — | — | — | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / Intel NPU | compile fails: the graph's input shapes are dynamic (`Upper bounds are not specified for node ...`) | — | — | — | — | — |
| int4 (RTN, CPU recipe), enable_causallm | GenAI 0.19.0 | OpenVINO / Intel GPU, NPU | session fails with every session or only the decoder on the device: `the model graph has been partitioned into 277 subgraphs` (enable_causallm needs one) | — | — | — | — | — |
<!-- end of table -->

## After the Windows update

The laptop installed a Windows update and restarted between 17:31 and 17:35 on 11-06 (build 26200.6899 to 26200.9711, `results/r8_os_builds.txt`). Foundry Local, the CUDA DLLs and the GPU driver stayed the same. Every step of this README ran once on the laptop, in order, from an empty folder: steps 1 to 4 before the update and steps 5 to 11 after it (`results/r8_walk_block1.log` to `results/r8_walk_block12.log`, one log per code block).

- Every Foundry Local form of steps 5 to 8 gives the same answers, word for word, wherever both runs answered (`results/r8_answers_vs_before.txt`). The two 131072-token forms that ran out of GPU memory before now answer all 24. The Hugging Face reference made again in step 9 gives the same 24 answers.
- On a quiet PC, the 8K int4 package decodes at 32 tokens/s instead of 58.8 (`results/r8_diag_after_update.txt`). With the daemon limited to the four P-cores (processor affinity) it decodes at 40 to 42. Its decode follows the daemon's CPU placement, and one GPU reading per run showed 21 to 43 % utilization, so the limit is on the CPU side. What the update changed there is not established.
- The CPU package on Foundry Local's CPU EP decodes at 9.1 tokens/s with 33,072 ms to the first token, against 11.6 and 20,195 ms before, in the same procedure (`results/r8_cpu_after.txt`).
- In ONNX Runtime GenAI 0.19.0 (steps 10 and 11), every form gives the same texts and first tokens as before. On CUDA without CUDA graphs, decode takes 7 to 9 ms more per token: int4 runs at 36.5 tokens/s instead of 54.7, `mixed` at 37.5 instead of 50.7. With CUDA graphs the difference is under 1 ms per token: int4 at 98.0 instead of 103.8, `mixed` at 67.9 instead of 70.5 (`results/r8_walk_block11.log`).
- In step 11, WebGPU on the RTX decodes at 18.4 tokens/s instead of 57.1, with about the same prefill time (1,776 ms against 1,748) and GPU memory (7,875 MiB against 7,879). The CPU EP decodes at 10.2 instead of 15.6, and its 512-token prefill takes 18.7 s instead of 9.4. OpenVINO decodes at 6.4 instead of 7.9 on the Intel GPU and at 5.9 as before on the CPU, where its prefill takes 12.4 s instead of 5.8. WebGPU on the Intel GPU decodes at 17.4 instead of 18.9 (`results/r8_walk_block12.log`).
- The 131072-token int4 package rises only to 8,230 MiB (8,136 after load) over the 24 prompts, against 15,997 MiB before, then to 13,350 MiB in the speed run, with 249 ms to the first token. The `mixed` package answers all 24 prompts and peaks at 15,971 MiB in the speed run.
- Foundry Local's OpenVINO EP still fails to load the CPU package on the Intel GPU, the NPU and the CPU, with the same error (`results/r8_ov_recheck.txt`).
- Measured again on 11-09, with MSI Center's User Scenario on Extreme Performance (set that evening) and Windows' power mode on Best performance: WebGPU on the Intel GPU, started from an ssh session while the desktop is locked, decodes at 19.3 tokens/s with High QoS (see Results), within 5 % of the 18.9 before the update, and at 18.3 without it (two processes each, `results/speed_high_qos_1109_webgpu_igpu.txt`). The numbers in the bullets above were measured on 11-06 with MSI Center's earlier setting, not Extreme Performance.
- Measured again on 11-11, with MSI Center's User Scenario on Balanced and Windows' power mode on its default: WebGPU on the RTX, started from an ssh session while the desktop is locked, decodes at 59.0 tokens/s with High QoS (see Results), within 5 % of the 57.1 before the update, and at 18.6 without it (two processes each, `results/speed_high_qos_1111_balanced.txt`).
- Measured again later on 11-11, with MSI Center's User Scenario on Balanced and Windows' power mode on its default, every process started from an ssh session while the desktop was locked (each number is the mean of two processes, `results/speed_high_qos_1111_balanced_2.txt`): the 8K int4 package in Foundry Local decodes at 61.1 tokens/s with High QoS set on the daemon's process by another process, within 5 % of the 58.8 before the update, and at 32.4 without it. ONNX Runtime GenAI's CPU EP decodes at 16.3 with High QoS, within 5 % of the 15.6 before the update, and at 10.5 without it. OpenVINO on the Intel GPU, in the second of its two runs (the file gives the first), decodes at 8.35 with High QoS and at 6.35 without it, against 7.9 before the update and 6.4 after it. The control of that file, a fixed graph timed on the same GPU right before and after those OpenVINO runs, took about 7 % less time than its usual 98.2 ms.

## Files

| File | What it does |
| --- | --- |
| `export_e4b.py` | Mobius export: `mobius.build`, `ModelPackage.save`, `write_ort_genai_config` (the calls of Olive's `MobiusBuilder` pass) |
| `quant_e4b.py` | Olive quantization per component: the RTN settings of the olive-recipes E2B `int4-optimized` recipes (without their graph surgery step), or `--recipe mixed` |
| `mixed_config.py` | The int8 node list of the `mixed` recipe, read from the E4B decoder graph |
| `olive_run_chunked.py` | Runs an Olive config with the k-quant search done in chunks of groups: the same output with a bounded peak memory |
| `variant_pkg.py` | A variant of a package: hard links to the weights and a new `genai_config.json` (where each session runs, context length, CUDA graphs) |
| `hf_ref.py` | The Hugging Face reference: bf16, greedy, 48 new tokens, first-token logits |
| `genai_parity.py` | A package in ONNX Runtime GenAI against the reference: first token, teacher forcing, greedy text |
| `genai_speed.py` | Prefill and decode time in ONNX Runtime GenAI |
| `fl_rest.py` | The 24 prompts and the speed run over Foundry Local's REST API |
| `score_keys.py` | The answer keys: a form passes when it answers right wherever the reference does, with at most one miss |
| `make_table.py` | The results table above, from `results/` |
| `make_figure.py` | The figure at the top of the README, from the same files |
| `requirements-*.txt` | The Python packages of each environment: export, reference, GenAI with CUDA, GenAI with the CPU / OpenVINO / WebGPU plugins |
| `fixture/` | The 24 prompts and the answer keys, written before any ONNX output was measured |
| `results/` | Every file behind the table: answers, speed runs, the Foundry Local logs (GPU memory, loaded DLLs, errors). `r8_*`: the runs after the Windows update, with one log per code block of the steps |
