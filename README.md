# Gemma 4 E4B as a Foundry Local model on Windows

[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) is not in the Foundry Local catalog ([microsoft/Foundry-Local#1232](https://github.com/microsoft/Foundry-Local/issues/1232) asks for it; the catalog has Gemma 4 E2B). This repo builds ONNX Runtime GenAI packages of E4B with [Mobius](https://github.com/onnxruntime/mobius), the exporter that the [olive-recipes Gemma 4 E2B recipes](https://github.com/microsoft/olive-recipes/tree/main/google-gemma-4-E2B-it) use, and runs them as local models in Foundry Local 0.13.0 on one Windows laptop.

The checks here are text in, text out. The packages also carry the vision and audio encoders; they load, but no image or audio prompt was checked.

Machine: MSI Stealth 16 AI+, Windows 11 build 26200, Core Ultra 9 386H, 31.5 GB RAM, RTX 5080 Laptop 16 GB (driver 591.74), Intel Graphics (driver 32.0.101.8724), Intel AI Boost NPU (driver 32.0.100.4841). Gemma 4 E4B at revision `4ecc2ba63b`.

## Results

One laptop, 2026-11-06. The 24 prompts in `fixture/` run greedy with 48 new tokens, against Hugging Face transformers 5.22.1 in bf16 on the same machine (`results/hf_ref.json`). The answer keys were written before any ONNX output was measured. A form passes when it answers right wherever the reference does (20 of the 24 prompts), with at most one miss. "Text = HF" counts answers identical to the reference's text. "First token = HF" counts the same first token, with the largest logit difference (GenAI only: Foundry Local's REST API returns text).

Speed: Foundry Local = a streamed chat completion over REST with a 933-token prompt and 160 new tokens; GenAI 0.19.0 = a 512-token prompt and 128 new tokens. Each is the median of 3 runs after a warm-up; the two runners used different prompts, so compare within one runner. VRAM: for Foundry Local, the daemon's own dedicated GPU memory (Windows performance counter); for GenAI, nvidia-smi's total.

<!-- table: python make_table.py --public -->
| Form | Runner | EP / device | Keys (bar) | Text = HF | First token = HF | Prefill | Decode | VRAM peak |
|---|---|---|---|---|---|---|---|---|
| fp16, embedding session on CPU | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 22/24 | — | TTFT 350 ms (933 tokens) | 29.6 tok/s | 12,630 MiB |
| fp16, embedding session on CPU | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 22/24 | 24/24 (max abs Δ 1.1) | — | — | — |
| int4 (RTN) | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 265 ms (933 tokens) | 59.4 tok/s | 14,040 MiB |
| int4 (RTN) | GenAI 0.19.0 | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.4) | 131 ms (512 tokens) | 51.7 tok/s | 10,316 MiB |
| int4 (RTN), embedding on CPU, 8K context | Foundry Local | CUDA / RTX 5080 Laptop | 20/20 PASS | 13/24 | — | TTFT 421 ms (933 tokens) | 13.5 tok/s | 9,182 MiB |
| int4 (RTN, CPU recipe) | Foundry Local | CPU / Core Ultra 9 386H | 20/20 PASS | 13/24 | — | — | — | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | CPU / Core Ultra 9 386H | 20/20 PASS | 13/24 | 22/24 (max abs Δ 33.8) | 9,388 ms (512 tokens) | 15.6 tok/s | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / CPU | 20/20 PASS | 13/24 | 22/24 (max abs Δ 33.9) | 5,785 ms (512 tokens) | 5.9 tok/s | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / Intel GPU | 20/20 PASS | 13/24 | 22/24 (max abs Δ 34.2) | 2,178 ms (512 tokens) | 7.9 tok/s | — |
| int4 (RTN, CPU recipe) | Foundry Local | OpenVINO / Intel GPU | load fails: `OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build` (Foundry-Local#1261) | — | — | — | — | — |
| int4 (RTN, CPU recipe) | GenAI 0.19.0 | OpenVINO / Intel NPU | compile fails: the graph's input shapes are dynamic (`Upper bounds are not specified for node ...`) | — | — | — | — | — |
<!-- end of table -->

## Steps

All steps run in PowerShell from one empty folder. Python 3.12 (`py -3.12`) and Foundry Local 0.13.0 (`winget install Microsoft.FoundryLocal`) must be installed.

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
# 3. Foundry Local's CUDA EP loads the CUDA 12 runtime and cuDNN 9 from PATH; it does not download them
py -3.12 -m venv .venv-dll
& .venv-dll\Scripts\python.exe -m pip install --target cuda12 --no-deps nvidia-cuda-runtime-cu12==12.9.79 nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-cublas-cu12==12.9.2.10 nvidia-cudnn-cu12==9.28.0.42
New-Item -ItemType Directory -Force cuda12-dll | Out-Null
Get-ChildItem cuda12 -Recurse -Filter *.dll | Copy-Item -Destination cuda12-dll
$env:Path = "$PWD\cuda12-dll;$env:Path"
foundry server stop
foundry server start
```

```powershell
# 4. Register the folder as a local model (no copy) and ask one question
foundry model register (Resolve-Path e4b-cuda-int4) gemma-4-e4b-it-cuda-int4:1 --task vision-language-chat --device gpu --execution-provider CUDAExecutionProvider
foundry model load gemma-4-e4b-it-cuda-int4:1
foundry complete gemma-4-e4b-it-cuda-int4:1 "What is the capital of Japan? Answer in one word." -m 48 -t 0
```

```powershell
# 5. The 24 prompts over the daemon's REST API, against the Hugging Face model's answers in results\hf_ref.json
$base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
$py = ".venv-export\Scripts\python.exe"
& $py fl_rest.py fixture "$base/catalogs/local/v1" gemma-4-e4b-it-cuda-int4:1 results\hf_ref.json fixture-cuda-int4.json
& $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json fixture-cuda-int4.json
foundry model unload gemma-4-e4b-it-cuda-int4:1
```

```powershell
# 6. The catalog E2B's decoder form for E4B: k-quant int4 with the MLP, o_proj and lm_head MatMuls at int8
#    (olive-recipes google-gemma-4-E2B-it/cuda/mixed); the other components are the int8 ones from step 2
$py = ".venv-export\Scripts\python.exe"
& $py mixed_config.py e4b-cuda-fp16\decoder\model.onnx mixed_upcast.json
& $py quant_e4b.py --src e4b-cuda-fp16 --dst e4b-cuda-mixed --target cuda --components decoder --recipe mixed --mixed_config mixed_upcast.json
foreach ($c in "embedding", "vision_encoder", "audio_encoder") { Copy-Item -Recurse e4b-cuda-int4\$c e4b-cuda-mixed\$c }
$env:Path = "$PWD\cuda12-dll;$env:Path"
foundry model register (Resolve-Path e4b-cuda-mixed) gemma-4-e4b-it-cuda-mixed:1 --task vision-language-chat --device gpu --execution-provider CUDAExecutionProvider
foundry model load gemma-4-e4b-it-cuda-mixed:1
$base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
& $py fl_rest.py fixture "$base/catalogs/local/v1" gemma-4-e4b-it-cuda-mixed:1 results\hf_ref.json fixture-cuda-mixed.json
& $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json fixture-cuda-mixed.json
foundry model unload gemma-4-e4b-it-cuda-mixed:1
```

```powershell
# 7. The CPU package: fp32 saved one component per process (the fp32 package is about 33 GB), then int4 / int8 weights
$py = ".venv-export\Scripts\python.exe"
foreach ($c in "decoder", "embedding", "vision_encoder", "audio_encoder") {
  & $py export_e4b.py --model gemma-4-E4B-it --hf_id google/gemma-4-E4B-it --ep cpu --dtype f32 --out e4b-cpu-fp32 --components $c
}
& $py export_e4b.py --model gemma-4-E4B-it --hf_id google/gemma-4-E4B-it --ep cpu --dtype f32 --out e4b-cpu-fp32 --config_only
& $py quant_e4b.py --src e4b-cpu-fp32 --dst e4b-cpu-int4 --target cpu
foundry model register (Resolve-Path e4b-cpu-int4) gemma-4-e4b-it-cpu-int4:1 --task vision-language-chat --device cpu --execution-provider CPUExecutionProvider
foundry model load gemma-4-e4b-it-cpu-int4:1
$base = (Get-Content "$env:USERPROFILE\.foundry\daemon.json" -Raw | ConvertFrom-Json).web_urls[0]
& $py fl_rest.py fixture "$base/catalogs/local/v1" gemma-4-e4b-it-cpu-int4:1 results\hf_ref.json fixture-cpu-int4.json
& $py score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json fixture-cpu-int4.json
foundry model unload gemma-4-e4b-it-cpu-int4:1
```

```powershell
# 8. The CPU package on the Intel GPU through the OpenVINO execution provider, outside Foundry Local (see below)
py -3.12 -m venv .venv-openvino
& .venv-openvino\Scripts\python.exe -m pip install -r requirements-genai-openvino.txt
& .venv-openvino\Scripts\python.exe genai_parity.py -m e4b-cpu-int4 --provider openvino --ov_device GPU --ref results\hf_ref.json --out parity-openvino-gpu.json --skip_forced
& .venv-export\Scripts\python.exe score_keys.py --keys fixture\answer_keys.json --ref results\hf_ref.json parity-openvino-gpu.json
```

## What to know

- Mobius 0.1.0 reads `config.head_dim`, which transformers 5.15.0 and later refuse for Gemma 4 (`AmbiguousGlobalPerLayerAttributeError`, [huggingface/transformers#47384](https://github.com/huggingface/transformers/pull/47384)). `requirements-export.txt` pins transformers 5.14.1. Mobius main reads it the new way.
- `write_ort_genai_config` takes the tokenizer from the Hub by model id; a local folder as the id fails with `HFValidationError`. `export_e4b.py --hf_id` passes the Hub id, as Olive's `MobiusBuilder` does.
- olive-ai 0.13.0 imports `requests` without installing it; `requirements-export.txt` lists it.
- onnx refuses external data that has more than one hard link (`may indicate a hard link attack`): quantize a package before making hard-linked variants of it.
- Registered local models answer on the daemon's `/catalogs/local/v1` path. The `/v1` path answers HTTP 404 `Model not found` for them.
- The fp16 package is 16.6 GB and does not fit in a 16 GB GPU. `variant_pkg.py --place embedding=cpu` runs the embedding session (a lookup) on the CPU.
- Foundry Local 0.13.0's own OpenVINO execution provider (the WinML plugin package 1.8.95.0) fails to load these packages with `OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build` ([microsoft/Foundry-Local#1261](https://github.com/microsoft/Foundry-Local/issues/1261)). Step 8 runs the CPU package on the Intel GPU through ONNX Runtime GenAI 0.19.0 and the pip plugin `onnxruntime-ep-openvino` 1.8.1 instead. The Intel NPU does not compile this graph: its input shapes are dynamic (`Upper bounds are not specified for node ...`).
- The int4 package keeps 9.3 GB of GPU memory after `foundry model load` and 14.0 GB after a 933-token prompt, with the 131072-token context of the export. `variant_pkg.py --max_length 8192 --place embedding=cpu` makes a variant with an 8K context and the embedding on the CPU.
- The text of the int4 (RTN) package differs from the Hugging Face model's on more prompts than fp16 does (13/24 equal against 22/24), while the keyed answers stay right. The gap comes from the int4 decoder weights, not from the int8 embedding: an int4 decoder with the fp16 embedding gives the same 13/24, and the fp16 decoder with the int8 embedding gives 22/24 (`results/parity_diag_*.json`).
- `foundry cache cd` asks to confirm a daemon restart and fails in a non-interactive shell; `--force` skips only that question.

## Files

| File | What it does |
| --- | --- |
| `export_e4b.py` | Mobius export: `mobius.build`, `ModelPackage.save`, `write_ort_genai_config` (the calls of Olive's `MobiusBuilder` pass) |
| `quant_e4b.py` | Olive quantization per component: the RTN settings of the olive-recipes E2B `int4-optimized` recipes (without their graph surgery step), or `--recipe mixed` |
| `mixed_config.py` | The int8 node list of the `mixed` recipe, read from the E4B decoder graph |
| `olive_run_chunked.py` | Runs an Olive config with the k-quant search done in chunks of groups: the same output with a bounded peak memory |
| `variant_pkg.py` | A variant of a package: hard links to the weights and a new `genai_config.json` (where each session runs, context length) |
| `hf_ref.py` | The Hugging Face reference: bf16, greedy, 48 new tokens, first-token logits |
| `genai_parity.py` | A package in ONNX Runtime GenAI against the reference: first token, teacher forcing, greedy text |
| `genai_speed.py` | Prefill and decode time in ONNX Runtime GenAI |
| `fl_rest.py` | The 24 prompts and the speed run over Foundry Local's REST API |
| `score_keys.py` | The answer keys: a form passes when it answers right wherever the reference does, with at most one miss |
| `make_table.py` | The results table above, from `results/` |
| `fixture/` | The 24 prompts and the answer keys, written before any ONNX output was measured |
| `results/` | Every file behind the table |

## License

The scripts are under the Apache License 2.0 (`LICENSE`). Gemma 4 is under the Apache License 2.0 ([model card](https://huggingface.co/google/gemma-4-E4B-it)).
