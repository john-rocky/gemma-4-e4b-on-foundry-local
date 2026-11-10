# Steps

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
