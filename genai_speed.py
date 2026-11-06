"""Prefill and decode speed of an ONNX Runtime GenAI package (Gemma 4 E4B).

Fixed prompt of N tokens (<bos> + a fixed English text), D new tokens forced with min_length, W warm-up runs,
then R timed runs. prefill = time of append_tokens(prompt); decode = D / time of D generate_next_token() calls.
GPU memory: nvidia-smi polled every 0.25 s while the runs go (peak MiB used on the RTX).

    python genai_speed.py -m <package dir> --provider cuda|cpu|openvino|webgpu [--ov_device GPU] --label <name> --out speed.jsonl
"""

import argparse
import json
import statistics
import subprocess
import threading
import time

import onnxruntime_genai as og

TEXT = ("The lighthouse keeper climbed the spiral stairs every evening, counting the steps as his father had taught him. "
        "From the lamp room he could see the harbor, the fishing boats returning with their nets, and the long line of the coast "
        "fading into the fog. He wrote the weather in a worn notebook: wind from the west, waves moderate, visibility falling. ")


class GpuPeak:
    def __init__(self):
        self.peak = 0
        self.base = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.peak = max(self.peak, value)
            time.sleep(0.25)

    def start(self):
        self.base = self._read()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def register_plugins():
    """Register the plugin EPs installed in this venv (OpenVINO, WebGPU), so a genai_config that names them loads."""
    import importlib

    for name in ("onnxruntime_ep_openvino", "onnxruntime_ep_webgpu"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        og.register_execution_provider_library(module.get_ep_name(), module.get_library_path())


def uses_webgpu(args):
    """True when a session runs on the WebGPU EP (by --provider or by the package's genai_config.json)."""
    if args.provider == "webgpu":
        return True
    if args.provider != "follow_config":
        return False
    import os

    with open(os.path.join(args.model, "genai_config.json"), encoding="utf-8") as f:
        return '"WebGPU"' in f.read()


def make_model(args):
    config = og.Config(args.model)
    if args.provider == "follow_config":
        register_plugins()
    if args.provider == "openvino":
        import onnxruntime_ep_openvino as ov_ep

        og.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
        config.clear_providers()
        config.append_provider("OpenVINO")
        config.set_provider_option("OpenVINO", "device_type", args.ov_device)
    elif args.provider == "cuda":
        config.clear_providers()
        config.append_provider("cuda")
    elif args.provider == "webgpu":
        import onnxruntime_ep_webgpu as webgpu_ep

        og.register_execution_provider_library(webgpu_ep.get_ep_name(), webgpu_ep.get_library_path())
        config.clear_providers()
        config.append_provider("WebGPU")
    elif args.provider == "cpu":
        config.clear_providers()
    names = {"openvino": "OpenVINO", "webgpu": "WebGPU"}
    for item in args.ep_option:
        key, value = item.split("=", 1)
        config.set_provider_option(names.get(args.provider, args.provider), key, value)
    if args.overlay:
        config.overlay(args.overlay)
    return og.Model(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--provider", default="follow_config", choices=["follow_config", "cuda", "cpu", "openvino", "webgpu"])
    parser.add_argument("--ov_device", default="CPU")
    parser.add_argument("--ep_option", action="append", default=[])
    parser.add_argument("--overlay", default="")
    parser.add_argument("--prompt_tokens", type=int, default=512)
    parser.add_argument("--new", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gpu = GpuPeak()
    gpu.start()
    t0 = time.perf_counter()
    model = make_model(args)
    load_s = time.perf_counter() - t0
    tokenizer = og.Tokenizer(model)
    body = [int(i) for i in tokenizer.encode(TEXT * 40)]
    if body and body[0] == 2:
        body = body[1:]
    ids = [2] + body[: args.prompt_tokens - 1]
    rows = []
    for run in range(args.warmup + args.runs):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + args.new, min_length=len(ids) + args.new, do_sample=False)
        g = og.Generator(model, params)
        t = time.perf_counter()
        g.append_tokens(ids)
        t1 = time.perf_counter()
        for _ in range(args.new):
            g.generate_next_token()
        t2 = time.perf_counter()
        del g
        row = {"label": args.label, "run": run, "warmup": run < args.warmup, "prompt_tokens": len(ids), "new_tokens": args.new,
               "prefill_ms": (t1 - t) * 1000, "decode_ms": (t2 - t1) * 1000, "decode_tok_s": args.new / (t2 - t1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    gpu.stop()
    timed = [r for r in rows if not r["warmup"]]
    summary = {"label": args.label, "summary": True, "model": args.model, "provider": args.provider,
               "ov_device": args.ov_device if args.provider == "openvino" else "", "ep_option": args.ep_option, "overlay": args.overlay,
               "og": og.__version__, "load_s": load_s, "prompt_tokens": len(ids), "new_tokens": args.new, "runs": len(timed),
               "prefill_ms_median": statistics.median(r["prefill_ms"] for r in timed),
               "prefill_tok_s_median": len(ids) / (statistics.median(r["prefill_ms"] for r in timed) / 1000),
               "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in timed),
               "decode_ms_per_token_median": statistics.median(r["decode_ms"] / args.new for r in timed),
               "gpu_mem_before_mib": gpu.base, "gpu_mem_peak_mib": gpu.peak}
    print("SUMMARY", json.dumps(summary), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summary) + "\n")
    if uses_webgpu(args):
        # The WebGPU plugin 0.5.0 on the RTX adapter kept the process alive after main() returned (outputs written, no
        # CPU use, 2026-11-06): leave without the interpreter's teardown once everything is on disk.
        import os
        import sys

        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
