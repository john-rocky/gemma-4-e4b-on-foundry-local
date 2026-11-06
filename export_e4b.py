"""Export google/gemma-4-E4B-it to an ONNX Runtime GenAI package with Mobius.

The same three calls as Olive's MobiusBuilder pass (olive/passes/onnx/mobius_model_builder.py, Olive 0.13.0), which the
olive-recipes google-gemma-4-E2B-it recipes use: mobius.build -> ModelPackage.save -> write_ort_genai_config.

    python export_e4b.py --model <local HF dir or id> --ep cuda --dtype f16 --out <dir>
"""

import argparse
import json
import os
import platform
import threading
import time


def rss_watch(stop, peak):
    try:
        import psutil
    except ImportError:
        return
    proc = psutil.Process(os.getpid())
    while not stop.is_set():
        try:
            peak[0] = max(peak[0], proc.memory_info().rss)
        except Exception:
            pass
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep", required=True, help="default|cpu|cuda|webgpu (mobius EP names)")
    parser.add_argument("--dtype", required=True, help="f32|f16|bf16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hf_id", default=None,
                        help="Hub id for the tokenizer / processor files and token ids (Olive passes the Hub id; a local folder path "
                             "fails in huggingface_hub with HFValidationError). Default: --model")
    parser.add_argument("--config_only", action="store_true",
                        help="build (lazy, under a minute) and write only the GenAI config files into --out; the graphs are already saved")
    parser.add_argument("--components", default=None,
                        help="comma list (decoder,embedding,vision_encoder,audio_encoder): save only these and skip the config. "
                             "One process per component keeps the peak memory to one component (an f32 E4B package is about 33 GB)")
    args = parser.parse_args()

    import mobius
    from mobius import build
    from mobius.integrations.ort_genai import write_ort_genai_config

    stop, peak = threading.Event(), [0]
    watcher = threading.Thread(target=rss_watch, args=(stop, peak), daemon=True)
    watcher.start()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    pkg = build(args.model, dtype=args.dtype, execution_provider=args.ep, load_weights=True)
    t1 = time.perf_counter()
    print(f"built components {list(pkg.keys())} in {t1 - t0:.0f} s, peak rss {peak[0] / 2**30:.1f} GiB", flush=True)
    wanted = set(args.components.split(",")) if args.components else None
    if not args.config_only:
        if wanted:
            # ModelPackage.save writes a package of one model straight into the folder as model.onnx (not into
            # <folder>/<name>/), so each component is saved into its own subfolder explicitly.
            for name in sorted(wanted):
                pkg.save(os.path.join(args.out, name), components=lambda n, name=name: n == name)
        else:
            pkg.save(args.out)
    t2 = time.perf_counter()
    print(f"saved in {t2 - t1:.0f} s, peak rss {peak[0] / 2**30:.1f} GiB", flush=True)
    artifacts = [] if wanted else write_ort_genai_config(pkg, args.out, hf_model_id=args.hf_id or args.model, ep=args.ep)
    stop.set()
    info = {"model": args.model, "hf_id": args.hf_id or args.model, "config_only": args.config_only, "ep": args.ep, "dtype": args.dtype,
            "mobius": getattr(mobius, "__version__", "?"),
            "components": list(pkg.keys()), "genai_artifacts": sorted(artifacts), "build_s": t1 - t0, "save_s": t2 - t1,
            "peak_rss_gib": peak[0] / 2**30, "python": platform.python_version()}
    info["saved_components"] = sorted(wanted) if wanted else ("none" if args.config_only else "all")
    suffix = ("_" + "_".join(sorted(wanted))) if wanted else ""
    with open(os.path.join(args.out, f"export_info{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
