"""Quantize an exported Gemma 4 E4B GenAI package component by component with Olive's OnnxBlockWiseRtnQuantization,
the pass and settings of olive-recipes google-gemma-4-E2B-it/{cuda,cpu}/int4-optimized (without the GraphSurgeries step,
which needs a patched Olive). Writes a new package; the source package is not touched.

    python quant_e4b.py --src <exported package> --dst <new package> --target cuda|cpu
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

SETTINGS = {
    # component -> pass parameters (cuda: recipe cuda/int4-optimized; cpu: recipe cpu/int4-optimized)
    "cuda": {
        "decoder": {"bits": 4, "block_size": 32, "is_symmetric": True},
        "embedding": {"bits": 8, "block_size": 32, "is_symmetric": False},
        "vision_encoder": {"bits": 8, "block_size": 32, "is_symmetric": False},
        "audio_encoder": {"bits": 8, "block_size": 32, "is_symmetric": False},
    },
    "cpu": {
        "decoder": {"bits": 4, "block_size": 32, "is_symmetric": True, "accuracy_level": 4},
        "embedding": {"bits": 8, "block_size": 128, "is_symmetric": True, "accuracy_level": 4},
        "vision_encoder": {"bits": 8, "block_size": 128, "is_symmetric": True, "accuracy_level": 4},
        "audio_encoder": {"bits": 8, "block_size": 128, "is_symmetric": True, "accuracy_level": 4},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--target", required=True, choices=["cuda", "cpu"])
    parser.add_argument("--components", default="decoder,embedding,vision_encoder,audio_encoder")
    parser.add_argument("--recipe", default="rtn", choices=["rtn", "mixed"],
                        help="mixed = the E2B cuda/mixed decoder recipe: OnnxKQuantQuantization at 4 bits with the MatMuls "
                             "named in --mixed_config at 8 bits (the other components keep the rtn settings)")
    parser.add_argument("--mixed_config", default="", help="node name -> {\"bits\": 8} (mixed_config.py)")
    parser.add_argument("--accuracy_level", type=int, default=-1,
                        help="override the MatMulNBits accuracy_level of every component (0 = unset: fp32 compute on the CPU; "
                             "4 = int8 compute, the cpu recipe value)")
    args = parser.parse_args()
    if args.recipe == "mixed" and not args.mixed_config:
        sys.exit("--recipe mixed needs --mixed_config")
    try:
        import olive  # noqa: F401  olive-ai 0.13.0 imports requests (telemetry) without declaring it
    except ImportError as error:
        sys.exit(f"Olive does not import in this Python: {error}")

    if not os.path.isfile(os.path.join(args.src, "genai_config.json")):
        sys.exit(f"--src {args.src} is not a package folder (no genai_config.json)")
    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):  # package-level files (genai_config.json, tokenizer, processors)
        path = os.path.join(args.src, name)
        if os.path.isfile(path) and not name.endswith((".onnx", ".onnx.data")):  # a graph belongs in its component folder
            shutil.copy2(path, os.path.join(args.dst, name))
    work = os.path.join(args.dst, "_olive")
    os.makedirs(work, exist_ok=True)
    report = {"src": args.src, "target": args.target, "components": {}}
    info_path = os.path.join(args.dst, "quant_info.json")
    if os.path.exists(info_path):  # a component that finished in an earlier run is kept, not quantized again
        with open(info_path, encoding="utf-8") as f:
            report["components"] = {k: v for k, v in json.load(f).get("components", {}).items()
                                    if isinstance(v, dict) and v.get("exit") == 0}
    for component in args.components.split(","):
        src_dir = os.path.join(args.src, component)
        if not os.path.isdir(src_dir):
            report["components"][component] = "missing in src"
            continue
        if component in report["components"] and os.path.exists(os.path.join(args.dst, component, "model.onnx")):
            print(component, "already done, kept", flush=True)
            continue
        params = dict(SETTINGS[args.target][component])
        if args.accuracy_level >= 0:
            params.pop("accuracy_level", None)
            if args.accuracy_level > 0:
                params["accuracy_level"] = args.accuracy_level
        params.update({"type": "OnnxBlockWiseRtnQuantization", "save_as_external_data": True, "external_data_name": "model.onnx.data"})
        if args.recipe == "mixed" and component == "decoder":
            with open(args.mixed_config, encoding="utf-8") as f:
                upcast = json.load(f)
            params = {"type": "OnnxKQuantQuantization", "bits": 4, "block_size": 32, "customized_weight_config": upcast,
                      "save_as_external_data": True, "external_data_name": "model.onnx.data"}
        out_dir = os.path.join(args.dst, component)
        config = {
            "input_model": {"type": "ONNXModel", "model_path": src_dir, "onnx_file_name": "model.onnx"},
            "passes": {"rtn": params},
            "output_dir": out_dir,
            "cache_dir": os.path.join(work, "cache_" + component),
            "log_severity_level": 1,
        }
        # No "target" block: the recipe names CUDAExecutionProvider for the cuda files, but this venv carries the CPU
        # onnxruntime and OnnxBlockWiseRtnQuantization does not read the target (olive/passes/onnx/rtn_quantization.py,
        # Olive 0.13.0), so the quantized graph is the same.
        cfg_path = os.path.join(work, component + ".json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=1)
        t = time.perf_counter()
        if params["type"] == "OnnxKQuantQuantization":
            # same pass, k-quant search over chunks of groups (identical output, bounded peak RAM): olive_run_chunked.py
            runner = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "olive_run_chunked.py"), cfg_path]
        else:
            runner = [sys.executable, "-m", "olive", "run", "--config", cfg_path]
        code = subprocess.call(runner)
        if code == 0:  # Olive's cache holds a second copy of the output model
            shutil.rmtree(os.path.join(work, "cache_" + component), ignore_errors=True)
        sizes = {n: os.path.getsize(os.path.join(out_dir, n)) for n in os.listdir(out_dir)} if os.path.isdir(out_dir) else {}
        report["components"][component] = {"exit": code, "s": round(time.perf_counter() - t, 1), "params": params, "files": sizes}
        print(component, json.dumps(report["components"][component]), flush=True)
    with open(os.path.join(args.dst, "quant_info.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    failed = [name for name, row in report["components"].items() if not isinstance(row, dict) or row["exit"] != 0]
    if failed:
        sys.exit(f"quantization failed for {', '.join(failed)} (see the Olive output above)")


if __name__ == "__main__":
    main()
