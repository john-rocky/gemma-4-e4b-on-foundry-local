# Gemma 4 E4B as a Foundry Local model on Windows

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

## Steps

Clone this repo into an empty folder and run the steps in PowerShell from it, in order. They need Python 3.12 (`py -3.12`), Foundry Local 0.13.0 (`winget install Microsoft.FoundryLocal`) and an NVIDIA driver for CUDA 12 and 13. After the last step the folder takes 99 GB of disk (`results/r8_walk_disk.txt`). Its files add up to 223 GB because the variants are hard links to their package's weight files.

```powershell
# 1. The weights (16 GB) and the export environment
py -3.12 -m venv .venv-export
$py = ".venv-export\Scripts\python.exe"
& $py -m pip install -r requirements-export.txt
& .venv-export\Scripts\hf.exe download google/gemma-4-E4B-it --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c --local-dir gemma-4-E4B-it
```

```powershell
# 2. Export for CUDA in fp16, then int4 weights for the decoder and int8 for the rest
$py = ".venv-export\Scripts\python.exe"
& $py export_e4b.py --model gemma-4-E4B-it --hf_id google/gemma-4-E4B-it --ep cuda --dtype f16 --out e4b-cuda-fp16
& $py quant_e4b.py --src e4b-cuda-fp16 --dst e4b-cuda-int4 --target cuda
```

```powershell
# 3. The catalog E2B's decoder form for E4B: k-quant int4 with the MLP, o_proj and lm_head MatMuls at int8
#    (olive-recipes google-gemma-4-E2B-it/cuda/mixed); the other components are the int8 ones from step 2.
#    Run it before step 7: the quantizer refuses weight files that have hard links.
$py = ".venv-export\Scripts\python.exe"
& $py mixed_config.py e4b-cuda-fp16\decoder\model.onnx mixed_upcast.json
& $py quant_e4b.py --src e4b-cuda-fp16 --dst e4b-cuda-mixed --target cuda --components decoder --recipe mixed --mixed_config mixed_upcast.json
foreach ($c in "embedding", "vision_encoder", "audio_encoder") { Copy-Item -Recurse e4b-cuda-int4\$c e4b-cuda-mixed\$c }
```

```powershell
# 4. Foundry Local's CUDA EP loads the CUDA 12 runtime and cuDNN 9 from PATH; it does not download them
py -3.12 -m venv .venv-dll
& .venv-dll\Scripts\python.exe -m pip install --target cuda12 --no-deps nvidia-cuda-runtime-cu12==12.9.79 nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-cublas-cu12==12.9.2.10 nvidia-cudnn-cu12==9.28.0.42
New-Item -ItemType Directory -Force cuda12-dll | Out-Null
Get-ChildItem cuda12 -Recurse -Filter *.dll | Copy-Item -Destination cuda12-dll
$env:Path = "$PWD\cuda12-dll;$env:Path"
foundry server stop
foundry server start
```

```powershell
# 5. The int4 package with an 8K-token context: the export sets 131072, and on a 16 GB GPU that slowed the daemon down
#    in a run before the Windows update (see "What to know"). variant_pkg.py makes a folder of hard links to a
#    package's weights with a new genai_config.json. Register that folder as a local model (no copy) and ask one question.
$py = ".venv-export\Scripts\python.exe"
& $py variant_pkg.py --src e4b-cuda-int4 --dst e4b-cuda-int4-8k --max_length 8192
$env:Path = "$PWD\cuda12-dll;$env:Path"
foundry model register (Resolve-Path e4b-cuda-int4-8k) gemma-4-e4b-it-cuda-int4-8k:1 --task vision-language-chat --device gpu --execution-provider CUDAExecutionProvider
foundry model load gemma-4-e4b-it-cuda-int4-8k:1
foundry complete gemma-4-e4b-it-cuda-int4-8k:1 "What is the capital of Japan? Answer in one word." -m 48 -t 0
```

```powershell
# 6. The 24 prompts over the daemon's REST API against the Hugging Face model's answers in results\hf_ref.json,
#    then the speed run (a 933-token prompt, 160 new tokens, 3 runs after a warm-up)
$base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
$py = ".venv-export\Scripts\python.exe"
& $py fl_rest.py fixture "$base/catalogs/local/v1" gemma-4-e4b-it-cuda-int4-8k:1 results\hf_ref.json fixture-e4b-cuda-int4-8k.json
& $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json fixture-e4b-cuda-int4-8k.json
& $py fl_rest.py speed "$base/catalogs/local/v1" gemma-4-e4b-it-cuda-int4-8k:1 speed-e4b-cuda-int4-8k.jsonl
foundry model unload gemma-4-e4b-it-cuda-int4-8k:1
```

```powershell
# 7. The other forms of the results table as local models: the export's 131072-token context and 32K, CUDA graphs,
#    fp16, the catalog E2B's mixed form, the embedding session on the CPU, the CUDA memory arena growing by request.
$py = ".venv-export\Scripts\python.exe"
& $py variant_pkg.py --src e4b-cuda-int4 --dst e4b-cuda-int4-32k --max_length 32768
& $py variant_pkg.py --src e4b-cuda-int4 --dst e4b-cuda-int4-graph --place decoder=cuda --cuda_option enable_cuda_graph=1
& $py variant_pkg.py --src e4b-cuda-fp16 --dst e4b-cuda-fp16-embcpu --place embedding=cpu
& $py variant_pkg.py --src e4b-cuda-fp16-embcpu --dst e4b-cuda-fp16-embcpu-32k --max_length 32768
& $py variant_pkg.py --src e4b-cuda-mixed --dst e4b-cuda-mixed-32k --max_length 32768
& $py variant_pkg.py --src e4b-cuda-mixed-32k --dst e4b-cuda-mixed-32k-graph --place decoder=cuda --cuda_option enable_cuda_graph=1
& $py variant_pkg.py --src e4b-cuda-int4 --dst e4b-cuda-int4-8k-emb8cpu --place embedding=cpu --max_length 8192
& $py variant_pkg.py --src e4b-cuda-int4 --dst e4b-cuda-int4-8k-embcpu --take embedding=e4b-cuda-fp16 --place embedding=cpu --max_length 8192
& $py variant_pkg.py --src e4b-cuda-int4-8k-embcpu --dst e4b-cuda-int4-8k-embcpu-arena --place decoder=cuda,vision=cuda,speech=cuda --cuda_option arena_extend_strategy=kSameAsRequested
& $py variant_pkg.py --src e4b-cuda-int4-8k-embcpu --dst e4b-cuda-int4-8k-embcpu-graph --place decoder=cuda --cuda_option enable_cuda_graph=1
# Each one: a fresh daemon (it keeps its GPU memory after an unload), register, load, the 24 prompts, the speed run.
# Before the Windows update, the 131072-token fp16 and mixed packages ran out of the 16 GB GPU on 18 of the 24 prompts
# in a run (results table); after it they answered all 24 ("After the Windows update").
$env:Path = "$PWD\cuda12-dll;$env:Path"
foreach ($m in @(@("e4b-cuda-int4", "gemma-4-e4b-it-cuda-int4:1"), @("e4b-cuda-int4-32k", "gemma-4-e4b-it-cuda-int4-32k:1"),
                 @("e4b-cuda-int4-graph", "gemma-4-e4b-it-cuda-int4-graph:1"), @("e4b-cuda-fp16-embcpu", "gemma-4-e4b-it-cuda-fp16-embcpu:1"),
                 @("e4b-cuda-fp16-embcpu-32k", "gemma-4-e4b-it-cuda-fp16-embcpu-32k:1"), @("e4b-cuda-mixed", "gemma-4-e4b-it-cuda-mixed:1"),
                 @("e4b-cuda-mixed-32k", "gemma-4-e4b-it-cuda-mixed-32k:1"), @("e4b-cuda-mixed-32k-graph", "gemma-4-e4b-it-cuda-mixed-32k-graph:1"),
                 @("e4b-cuda-int4-8k-emb8cpu", "gemma-4-e4b-it-cuda-int4-8k-emb8cpu:1"), @("e4b-cuda-int4-8k-embcpu", "gemma-4-e4b-it-cuda-int4-8k-embcpu:1"),
                 @("e4b-cuda-int4-8k-embcpu-arena", "gemma-4-e4b-it-cuda-int4-8k-embcpu-arena:1"),
                 @("e4b-cuda-int4-8k-embcpu-graph", "gemma-4-e4b-it-cuda-int4-8k-embcpu-graph:1"))) {
  $folder = $m[0]; $id = $m[1]
  foundry server stop
  foundry server start
  $base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
  foundry model register (Resolve-Path $folder) $id --task vision-language-chat --device gpu --execution-provider CUDAExecutionProvider
  foundry model load $id
  & $py fl_rest.py fixture "$base/catalogs/local/v1" $id results\hf_ref.json "fixture-$folder.json"
  & $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json "fixture-$folder.json"
  & $py fl_rest.py speed "$base/catalogs/local/v1" $id "speed-$folder.jsonl"
  foundry model unload $id
}
foundry server stop
```

```powershell
# 8. The CPU package: fp32 saved one component per process (the fp32 package is about 33 GB), then int4 / int8 weights
$py = ".venv-export\Scripts\python.exe"
foreach ($c in "decoder", "embedding", "vision_encoder", "audio_encoder") {
  & $py export_e4b.py --model gemma-4-E4B-it --hf_id google/gemma-4-E4B-it --ep cpu --dtype f32 --out e4b-cpu-fp32 --components $c
}
& $py export_e4b.py --model gemma-4-E4B-it --hf_id google/gemma-4-E4B-it --ep cpu --dtype f32 --out e4b-cpu-fp32 --config_only
& $py quant_e4b.py --src e4b-cpu-fp32 --dst e4b-cpu-int4 --target cpu
```

```powershell
# 8 (continued). The CPU package as a Foundry Local model on the CPU EP: the 24 prompts and the speed run
$py = ".venv-export\Scripts\python.exe"
$env:Path = "$PWD\cuda12-dll;$env:Path"
foundry server start
$base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
foundry model register (Resolve-Path e4b-cpu-int4) gemma-4-e4b-it-cpu-int4:1 --task vision-language-chat --device cpu --execution-provider CPUExecutionProvider
foundry model load gemma-4-e4b-it-cpu-int4:1
& $py fl_rest.py fixture "$base/catalogs/local/v1" gemma-4-e4b-it-cpu-int4:1 results\hf_ref.json fixture-cpu-int4.json
& $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json fixture-cpu-int4.json
& $py fl_rest.py speed "$base/catalogs/local/v1" gemma-4-e4b-it-cpu-int4:1 speed-cpu-int4.jsonl
foundry model unload gemma-4-e4b-it-cpu-int4:1
foundry server stop
```

```powershell
# 9. The Hugging Face reference again (bf16 on the GPU, the per-layer embedding table on the CPU): it writes the
#    first-token logits that steps 10 and 11 compare against, and its answers should equal results\hf_ref.json
py -3.12 -m venv .venv-ref
& .venv-ref\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
& .venv-ref\Scripts\python.exe -m pip install -r requirements-reference.txt
& .venv-ref\Scripts\python.exe hf_ref.py --model gemma-4-E4B-it --fixture fixture\prompts.json --out hf_ref.json --logits hf_first_logits.npy --device cuda
& .venv-ref\Scripts\python.exe -c "import json; a, b = (json.load(open(p, encoding='utf-8'))['records'] for p in ('hf_ref.json', 'results/hf_ref.json')); print(sum(x['text'] == y['text'] for x, y in zip(a, b)), 'of', len(b), 'answers equal results/hf_ref.json')"
```

```powershell
# 10. The CUDA packages in ONNX Runtime GenAI 0.19.0 from Python, outside Foundry Local: answers, first-token logits,
#     speed (a 512-token prompt, 128 new tokens, 3 runs after a warm-up)
py -3.12 -m venv .venv-genai-cuda
& .venv-genai-cuda\Scripts\python.exe -m pip install -r requirements-genai-cuda.txt
$sp = "$PWD\.venv-genai-cuda\Lib\site-packages"
$env:Path = "$sp\nvidia\cu13\bin\x86_64;$sp\nvidia\cu13\bin;$sp\nvidia\cudnn\bin;$env:Path"
$py = ".venv-genai-cuda\Scripts\python.exe"
$ref = "--ref", "hf_ref.json", "--ref_logits", "hf_first_logits.npy", "--skip_forced"
# int4 and mixed on CUDA, each without and with CUDA graphs
foreach ($m in "e4b-cuda-int4", "e4b-cuda-mixed") {
  & $py genai_parity.py -m $m --provider cuda @ref --out "parity-$m.json"
  & $py genai_speed.py -m $m --provider cuda --label $m --out "speed-genai-$m.jsonl"
  & $py genai_parity.py -m $m --provider cuda --ep_option enable_cuda_graph=1 @ref --out "parity-$m-graph.json"
  & $py genai_speed.py -m $m --provider cuda --ep_option enable_cuda_graph=1 --label "$m-graph" --out "speed-genai-$m-graph.jsonl"
}
# fp16 and the int8 embedding on the CPU, with the sessions where their genai_config.json puts them
# (GenAI 0.19.0's speed run stops on these two packages at the first decode step, see "What to know")
foreach ($m in "e4b-cuda-fp16-embcpu", "e4b-cuda-int4-8k-emb8cpu") {
  & $py genai_parity.py -m $m --provider follow_config @ref --out "parity-$m.json"
}
.venv-export\Scripts\python.exe score_keys.py --keys fixture\answer_keys.json --ref hf_ref.json parity-e4b-cuda-int4.json parity-e4b-cuda-int4-graph.json parity-e4b-cuda-mixed.json parity-e4b-cuda-mixed-graph.json parity-e4b-cuda-fp16-embcpu.json parity-e4b-cuda-int4-8k-emb8cpu.json
```

```powershell
# 11. ONNX Runtime GenAI 0.19.0 on the CPU, and with the OpenVINO and WebGPU execution provider plugins from pip
#     (Foundry Local 0.13.0 does not run these packages on the Intel GPU or the NPU, see "What to know")
py -3.12 -m venv .venv-genai
& .venv-genai\Scripts\python.exe -m pip install -r requirements-genai.txt
$py = ".venv-genai\Scripts\python.exe"
$ref = "--ref", "hf_ref.json", "--ref_logits", "hf_first_logits.npy", "--skip_forced"
# the CPU package on the CPU EP, then on the OpenVINO EP on the CPU and on the Intel GPU
& $py genai_parity.py -m e4b-cpu-int4 --provider cpu @ref --out parity-cpu-int4.json
& $py genai_speed.py -m e4b-cpu-int4 --provider cpu --label cpu-int4 --out speed-genai-cpu-int4.jsonl
foreach ($d in "CPU", "GPU") {
  & $py genai_parity.py -m e4b-cpu-int4 --provider openvino --ov_device $d @ref --out "parity-openvino-$d.json"
  & $py genai_speed.py -m e4b-cpu-int4 --provider openvino --ov_device $d --label "openvino-$d" --out "speed-genai-openvino-$d.jsonl"
}
# the CUDA int4 package with every session on WebGPU: adapter 0 is the RTX and adapter 1 the Intel GPU on this laptop
foreach ($a in "0", "1") {
  & $py variant_pkg.py --src e4b-cuda-int4 --dst "e4b-webgpu-int4-$a" --place decoder=webgpu,embedding=webgpu,vision=webgpu,speech=webgpu --webgpu_option adapterIndex=$a
  & $py genai_parity.py -m "e4b-webgpu-int4-$a" --provider follow_config @ref --out "parity-webgpu-$a.json"
  & $py genai_speed.py -m "e4b-webgpu-int4-$a" --provider follow_config --label "webgpu-$a" --out "speed-genai-webgpu-$a.jsonl"
}
.venv-export\Scripts\python.exe score_keys.py --keys fixture\answer_keys.json --ref hf_ref.json parity-cpu-int4.json parity-openvino-CPU.json parity-openvino-GPU.json parity-webgpu-0.json parity-webgpu-1.json
```

## What to know

The numbers in this section were measured before the Windows update of 11-06 17:31, unless a line says otherwise.

- Mobius 0.1.0 reads `config.head_dim`, which transformers 5.15.0 and later refuse for Gemma 4 (`AmbiguousGlobalPerLayerAttributeError`, [huggingface/transformers#47384](https://github.com/huggingface/transformers/pull/47384)). `requirements-export.txt` pins transformers 5.14.1. Mobius main reads it the new way.
- `write_ort_genai_config` takes the tokenizer from the Hub by model id; a local folder as the id fails with `HFValidationError`. `export_e4b.py --hf_id` passes the Hub id, as Olive's `MobiusBuilder` does.
- olive-ai 0.13.0 imports `requests` without installing it; `requirements-export.txt` lists it.
- onnx refuses external data that has more than one hard link (`may indicate a hard link attack`): quantize a package before making hard-linked variants of it.
- The `mixed` recipe's k-quant search runs on one CPU core: 27.5 minutes for the E4B decoder on this laptop.
- Registered local models answer on the daemon's `/catalogs/local/v1` path. The `/v1` path answers HTTP 404 `Model not found` for them (`results/v1_probe.txt`).
- Foundry Local's documented way to add a compiled model (a folder with `inference_model.json` in the model cache) did not list these packages with CLI 0.13.0, as a junction or as real folders in the catalog's layout (`Microsoft\<name>-1\v1\`) or flat; `foundry model register` works. While `foundry cache cd` points at another folder, the registered local models are not listed either.
- `foundry cache cd` asks to confirm a daemon restart and fails in a non-interactive shell; `--force` skips only that question.
- The daemon keeps a model's GPU memory after `foundry model unload`; `foundry server stop` frees it.
- The export sets a 131072-token context in `genai_config.json`. Before the Windows update, on the 16 GB GPU, the int4 package with that context grew from 8,132 MiB after load to 15,997 MiB over the 24 prompts. The speed run after them took 15 s to the first token and decoded at 43.3 tokens/s. As the only run in a fresh daemon, it took 265 ms and decoded at 59.4 (`results/r7_int4_s.txt`). With a 32K context it held 15,920 MiB at most and decoded at 53.6 tokens/s. With 8K it held 13,294 MiB at most and decoded at 58.8. After the update, the 131072-token package rose only to 8,230 MiB over the 24 prompts.
- The fp16 package with the 131072-token context answered all 24 prompts in one run (`results/r7_fp16_embcpu_b.txt`) and ran out of GPU memory on 18 of them in another (`results/r7_fp16_embcpu_full.txt`), both before the Windows update; after it, it answered all 24 in the README walk. With a 32K context it answered all 24 and decodes at 28.6 tokens/s, with 15,966 MiB at most (`results/r7_fp16_embcpu_32k.txt`).
- The fp16 package is 16.6 GB and does not fit in a 16 GB GPU. `variant_pkg.py --place embedding=cpu` runs the embedding session (a lookup) on the CPU.
- The int8 embedding runs slowly on the CPU: the 8K variant with it decodes at 13.5 tokens/s in Foundry Local and 14.3 in ONNX Runtime GenAI. `e4b-cuda-int4-8k-embcpu` takes the fp16 embedding from the fp16 package (`--take embedding=e4b-cuda-fp16`) and decodes at 47.0.
- `arena_extend_strategy=kSameAsRequested` in the CUDA options left the daemon's GPU memory as it was over the 24 prompts and the speed run: 4,042 MiB after load and 9,186 MiB at most, with and without it. As the only run in a fresh daemon, the daemon held 7,760 MiB at most without it and 7,796 MiB with it (46.6 and 48.4 tokens/s).
- The `mixed` decoder is 5.05 GB against 2.80 GB for RTN int4. Its first-token logits are closer to the Hugging Face model's (largest difference 16.3 against 34.4), and it gives the same 13 of 24 identical texts. It misses one keyed answer (p22: its 48 tokens stop before the answer).
- With the export's 131072-token context, the `mixed` package ran out of the 16 GB GPU on 18 of the 24 prompts (`CUDA error: out of memory`) in a run before the Windows update. After it, it answered all 24 and peaked at 15,971 MiB. With a 32K context it answers all 24, but the daemon holds 15,986 MiB, and the 933-token prompt takes 1.8 s to the first token after a 0.24 s warm-up.
- The text of the int4 (RTN) package differs from the Hugging Face model's on more prompts than fp16 does (13/24 equal against 22/24), while the keyed answers stay right. The gap comes from the int4 decoder weights: an int4 decoder with the fp16 embedding gives the same 13/24, and the fp16 decoder with the int8 embedding gives 22/24 (`results/parity_diag_*.json`). The CPU package with fp32 instead of int8 compute in its int4 MatMuls also gives 13/24, at 3.8 instead of 15.6 tokens/s (`results/parity_cpu_int4_acc0.json`).
- CUDA graphs (`enable_cuda_graph` 1 on the decoder session; the catalog E2B packages set 0) keep the answers and raise the decode speed in ONNX Runtime GenAI 0.19.0: int4 from 58.4 (measured again on 11-09, see Results) to 103.8 tokens/s, `mixed` from 50.7 to 70.5. In Foundry Local 0.13.0 the 8K and 32K variants keep their speed (47.0 to 47.8 and 49.3 to 49.7 tokens/s), and the 131072-token int4 package decodes at 23.5 instead of 43.3 tokens/s.
- ONNX Runtime GenAI 0.19.0's speed run stops at the first decode step on the packages whose embedding session runs on the CPU: `The output OrtValue provided for output 'inputs_embeds' of node 'InsertedPrecisionFreeCast_inputs_embeds' (Cast) has shape {1,512,2560} but the computed output shape for this run is {1,1,2560}`. It stopped in every run of the fp16 package and in some runs of the int8-embedding 8K package: the table's 14.3 tokens/s is from a run that went through, and `results/r8_walk_block11_first.log` has one that stopped. The 24 prompts run on both, and Foundry Local runs the same packages with the 933-token prompt.
- Foundry Local 0.13.0's own OpenVINO execution provider (the WinML plugin package 1.8.95.0) fails to load these packages on the Intel GPU, the NPU and the CPU alike, with `OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build` ([microsoft/Foundry-Local#1261](https://github.com/microsoft/Foundry-Local/issues/1261)). Step 11 runs the CPU package through ONNX Runtime GenAI 0.19.0 and the pip plugin `onnxruntime-ep-openvino` 1.8.1 instead: 7.9 tokens/s on the Intel GPU and 5.9 on the CPU, slower than the CPU EP (15.6).
- With `enable_causallm=True` (the provider option of the catalog's OpenVINO variants), the OpenVINO EP refuses this graph on the Intel GPU and the NPU, with every session or only the decoder on the device: `the model graph has been partitioned into 277 subgraphs`. The open olive-recipes PR [#527](https://github.com/microsoft/olive-recipes/pull/527) gives the cause for Gemma 4: OpenVINO's ONNX frontend has no `RMSNormalization` (opset 24; the E4B decoder has 235 of them) and fails shape inference on `Attention` with a KV cache. Without that option, the NPU does not compile the dynamic-shape graph: `Upper bounds are not specified for node ...`.
- Foundry Local 0.13.0 does not load a local model on the WebGPU EP: the load of a model registered with `--execution-provider WebGpuExecutionProvider` fails with `unknown execution provider for local model`. ONNX Runtime GenAI 0.19.0 with the pip plugin `onnxruntime-ep-webgpu` 0.5.0 runs the int4 package with every session on WebGPU: 18.9 tokens/s on the Intel GPU and 57.1 on the RTX, where CUDA decoded at 54.7 on 11-06 (58.4 when measured again on 11-09 with High QoS) and took 134 ms instead of 1,748 ms for the 512-token prompt. On the RTX, the Python process stayed alive after its run, so `genai_parity.py` and `genai_speed.py` exit right after writing their output when a session runs on WebGPU.

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
| `requirements-*.txt` | The Python packages of each environment: export, reference, GenAI with CUDA, GenAI with the CPU / OpenVINO / WebGPU plugins |
| `fixture/` | The 24 prompts and the answer keys, written before any ONNX output was measured |
| `results/` | Every file behind the table: answers, speed runs, the Foundry Local logs (GPU memory, loaded DLLs, errors). `r8_*`: the runs after the Windows update, with one log per code block of the steps |

## License

The scripts are under the Apache License 2.0 (`LICENSE`). Gemma 4 is under the Apache License 2.0 ([model card](https://huggingface.co/google/gemma-4-E4B-it)).
