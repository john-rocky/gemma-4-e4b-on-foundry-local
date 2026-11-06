"""Build the result table (markdown) from the files in results/.

    python make_table.py [--public]

Each row names its files; a missing file leaves the cell as "—". Keys are scored with score_keys.py against
fixture/answer_keys.json and results/hf_ref.json. --public prints the README form (no internal file names).
"""

import json
import os
import sys

import score_keys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# runner: "Foundry" (REST on the daemon, temperature 0, 48 tokens) or "GenAI" (onnxruntime-genai 0.19.0 in Python)
ROWS = [
    # form, runner, EP / device, fixture or parity file, speed file, Foundry log (VRAM peak)
    # RTX 5080 Laptop, CUDA EP, Foundry Local
    ("fp16, embedding session on CPU", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_fp16_embcpu_b.json", "fl_speed_fp16_embcpu_s.jsonl", "r7_fp16_embcpu_s.txt"),
    ("int4 (RTN)", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4.json", "fl_speed_int4_s.jsonl", "r7_int4_s.txt"),
    ("int4 (RTN), CUDA graphs", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_graph.json", "fl_speed_int4_graph.jsonl", "r7_int4_graph.txt"),
    ("int4 (mixed, catalog E2B recipe), 131072 context", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_mixed.json", "fl_speed_mixed_s.jsonl", "r7_mixed_s.txt"),
    ("int4 (mixed, catalog E2B recipe), 32K context", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_mixed_32k.json", "fl_speed_mixed_32k.jsonl", "r7_mixed_32k.txt"),
    ("int4 (mixed, catalog E2B recipe), 32K context, CUDA graphs", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_mixed_32k_graph.json", "fl_speed_mixed_32k_graph.jsonl", "r7_mixed_32k_graph.txt"),
    ("int4 (RTN), int8 embedding on CPU, 8K context", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_embcpu_8k.json", "fl_speed_int4_embcpu_8k.jsonl", "r7_int4_embcpu_8k.txt"),
    ("int4 (RTN), int8 embedding on CPU, 8K context, CUDA arena by request", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_embcpu_8k_arena.json", "fl_speed_int4_embcpu_8k_arena_s.jsonl", "r7_int4_embcpu_8k_arena_s.txt"),
    ("int4 (RTN), fp16 embedding on CPU, 8K context", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_embf16cpu_8k.json", "fl_speed_int4_embf16cpu_8k.jsonl", "r7_int4_embf16cpu_8k.txt"),
    ("int4 (RTN), fp16 embedding on CPU, 8K context, CUDA arena by request", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_embf16cpu_8k_arena.json", "fl_speed_int4_embf16cpu_8k_arena_s.jsonl", "r7_int4_embf16cpu_8k_arena_s.txt"),
    ("int4 (RTN), fp16 embedding on CPU, 8K context, CUDA graphs", "Foundry", "CUDA / RTX 5080 Laptop", "fl_fixture_int4_embf16cpu_8k_graph.json", "fl_speed_int4_embf16cpu_8k_graph.jsonl", "r7_int4_embf16cpu_8k_graph.txt"),
    # RTX 5080 Laptop, CUDA EP, GenAI direct
    ("fp16, embedding session on CPU", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_f16_embcpu.json", "speed_cuda_f16_embcpu_again.jsonl", None),
    ("int4 (RTN)", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_int4.json", "speed_cuda_int4.jsonl", None),
    ("int4 (RTN), CUDA graphs", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_int4_graph.json", "speed_cuda_int4_graph.jsonl", None),
    ("int4 (mixed, catalog E2B recipe)", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_mixed.json", "speed_cuda_mixed.jsonl", None),
    ("int4 (mixed, catalog E2B recipe), CUDA graphs", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_mixed_graph.json", "speed_cuda_mixed_graph.jsonl", None),
    ("int4 (RTN), int8 embedding on CPU, 8K context", "GenAI", "CUDA / RTX 5080 Laptop", "parity_cuda_int4_embcpu_8k.json", "speed_cuda_int4_embcpu_8k_again.jsonl", None),
    ("int4 (RTN), fp16 embedding on CPU", "GenAI", "CUDA / RTX 5080 Laptop", "parity_diag_e4b_dec4_embf16cpu.json", "speed_diag_e4b_dec4_embf16cpu.jsonl", None),
    # WebGPU (GenAI direct; Foundry Local's local models do not take the WebGPU EP, NOT_RUNNING below)
    ("int4 (RTN)", "GenAI", "WebGPU / RTX 5080 Laptop", "parity_webgpu_rtx_int4.json", "speed_webgpu_rtx_int4.jsonl", None),
    ("int4 (RTN)", "GenAI", "WebGPU / Intel GPU", "parity_webgpu_igpu_int4.json", "speed_webgpu_igpu_int4.jsonl", None),
    # OpenVINO (GenAI direct + the pip plugin; Foundry Local's own OpenVINO EP does not load, NOT_RUNNING below)
    ("int4 (RTN, CPU recipe)", "GenAI", "OpenVINO / Intel GPU", "parity_ov_gpu_cpu_int4.json", "speed_ov_gpu_cpu_int4.jsonl", None),
    ("int4 (RTN, CPU recipe), decoder only on the device, enable_causallm", "GenAI", "OpenVINO / Intel GPU", "parity_ov_gpu_dec.json", "speed_ov_gpu_dec.jsonl", None),
    ("int4 (RTN, CPU recipe), decoder only on the device, enable_causallm", "GenAI", "OpenVINO / Intel NPU", "parity_ov_npu_dec.json", "speed_ov_npu_dec.jsonl", None),
    ("int4 (RTN, CPU recipe)", "GenAI", "OpenVINO / CPU", "parity_ov_cpu_cpu_int4.json", "speed_ov_cpu_cpu_int4.jsonl", None),
    # CPU EP
    ("int4 (RTN, CPU recipe)", "Foundry", "CPU / Core Ultra 9 386H", "fl_fixture_cpu_int4.json", "fl_speed_cpu_int4_s.jsonl", None),
    ("int4 (RTN, CPU recipe)", "GenAI", "CPU / Core Ultra 9 386H", "parity_cpu_int4.json", "speed_cpu_int4.jsonl", None),
    ("int4 (RTN, CPU recipe), fp32 compute", "GenAI", "CPU / Core Ultra 9 386H", "parity_cpu_int4_acc0.json", "speed_cpu_int4_acc0.jsonl", None),
]

# rows kept out of the README table (--public): diagnostics and option checks, named in the README's notes instead
INTERNAL = {"parity_diag_e4b_dec4_embf16cpu.json", "fl_fixture_int4_embcpu_8k_arena.json", "parity_cpu_int4_acc0.json"}

# forms that do not run: (form, runner, EP / device, what happens, evidence file)
NOT_RUNNING = [
    ("int4 (RTN)", "Foundry", "WebGPU / Intel GPU",
     "load fails: `unknown execution provider for local model` (registered with `--execution-provider WebGpuExecutionProvider`)",
     "r7_webgpu_igpu.txt"),
    ("int4 (RTN, CPU recipe)", "Foundry", "OpenVINO / Intel GPU, NPU, CPU",
     "load fails on all three devices: `OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build` (Foundry-Local#1261)",
     "r7_ov_gpu.txt, r7_ov_npu.txt, r7_ov_cpu.txt"),
    ("int4 (RTN, CPU recipe)", "GenAI", "OpenVINO / Intel NPU",
     "compile fails: the graph's input shapes are dynamic (`Upper bounds are not specified for node ...`)", "r6_parity_ov_npu_cpu_int4.txt"),
    ("int4 (RTN, CPU recipe), enable_causallm", "GenAI", "OpenVINO / Intel GPU, NPU",
     "session fails: `the model graph has been partitioned into 277 subgraphs` (enable_causallm needs one)",
     "r6_parity_ov_gpu_causallm_probe.txt, r6_parity_ov_npu_causallm_probe.txt"),
]


def load(name):
    path = os.path.join(RES, name) if name else None
    if not path or not os.path.exists(path):
        return None
    if name.endswith(".jsonl"):
        lines = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        return next((x for x in reversed(lines) if x.get("summary")), None)  # the last run of a re-measured label
    return json.load(open(path, encoding="utf-8"))


def foundry_peak(log):
    """Highest 'daemon only' VRAM peak printed in an r7 Foundry log, in MiB."""
    path = os.path.join(RES, log) if log else None
    if not path or not os.path.exists(path):
        return None
    peaks = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if "daemon only:" in line:
            try:
                peaks.append(int(line.rsplit("daemon only:", 1)[1].strip(" )\n")))
            except ValueError:
                pass
    return max(peaks) if peaks else None


def main():
    public = "--public" in sys.argv
    keys = json.load(open(os.path.join(HERE, "fixture", "answer_keys.json"), encoding="utf-8"))["keys"]
    ref = json.load(open(os.path.join(RES, "hf_ref.json"), encoding="utf-8"))
    ref_ok = [r["id"] for r in ref["records"] if r["id"] in keys and score_keys.passes(keys[r["id"]], r["text"])]
    head = ["Form", "Runner", "EP / device", "Keys (bar)", "Text = HF", "First token = HF", "Prefill", "Decode", "VRAM peak"]
    if not public:
        head.append("Files")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for form, runner, ep, qual, speed, log in ROWS:
        if public and qual in INTERNAL:
            continue
        q, s = load(qual), load(speed)
        if q is None and s is None:
            continue
        cells = [form, "Foundry Local" if runner == "Foundry" else "GenAI 0.19.0", ep]
        if q is not None:
            rows = {r["id"]: r for r in q["records"]}
            miss = [p for p in ref_ok if p not in rows or "error" in rows[p] or not score_keys.passes(keys[p], rows[p].get("text"))]
            summ = q.get("summary", {})
            errs = [r["error"] for r in q["records"] if "error" in r]
            why = next((m for m in ("CUDA error: out of memory",) if any(m in e for e in errs)), "")
            failed = f"{len(errs)} of {len(q['records'])} requests failed" + (f": `{why}`" if why else "")
            if len(errs) * 2 > len(q["records"]):
                cells.append(f"not judged: {failed}")
            else:
                cells.append(f"{len(ref_ok) - len(miss)}/{len(ref_ok)} {'PASS' if len(miss) <= 1 else 'FAIL'}" + (f" ({failed})" if errs else ""))
            n = summ.get("n", len(q["records"]))
            eq = summ.get("equal", summ.get("greedy_equal"))
            cells.append(f"{eq}/{n}" if eq is not None else "—")
            if runner == "GenAI" and summ.get("first_top1") is not None:
                cells.append(f"{summ['first_top1']}/{n} (max abs Δ {summ['first_max_abs']:.1f})" if summ.get("first_max_abs") is not None else f"{summ['first_top1']}/{n}")
            else:
                cells.append("—")
        else:
            cells += ["—", "—", "—"]
        if s is not None and runner == "Foundry":
            cells.append(f"TTFT {s['ttft_ms_median']:,.0f} ms ({s['prompt_tokens']} tokens)")
            cells.append(f"{s['decode_tok_s_median']:.1f} tok/s")
            peak = foundry_peak(log) if "RTX" in ep else None
            cells.append(f"{peak:,} MiB" if peak else "—")
        elif s is not None:
            cells.append(f"{s['prefill_ms_median']:,.0f} ms ({s['prompt_tokens']} tokens)")
            cells.append(f"{s['decode_tok_s_median']:.1f} tok/s")
            peak = s.get("gpu_mem_peak_mib")
            cells.append(f"{peak:,} MiB" if peak and "RTX" in ep else "—")
        else:
            cells += ["—", "—", "—"]
        if not public:
            cells.append(", ".join(f"`{x}`" for x in (qual, speed, log) if x and os.path.exists(os.path.join(RES, x))))
        print("| " + " | ".join(cells) + " |")
    for form, runner, ep, what, evidence in NOT_RUNNING:
        files = [x.strip() for x in evidence.split(",")]
        if not all(os.path.exists(os.path.join(RES, x)) for x in files):
            continue
        cells = [form, "Foundry Local" if runner == "Foundry" else "GenAI 0.19.0", ep, what] + ["—"] * 5
        if not public:
            cells.append(", ".join(f"`{x}`" for x in files))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
