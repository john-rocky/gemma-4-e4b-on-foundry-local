"""Compare an ONNX Runtime GenAI package of Gemma 4 E4B with the Hugging Face reference (hf_ref.py).

Per prompt, on the reference's own token ids:
  first   - logits of the first new token over the full vocabulary: top-1 match and max |delta|
  forced  - teacher forcing along the reference's greedy tokens: top-1 match at every step, max |delta| on the reference's top-8
  greedy  - free greedy decoding: tokens equal to the reference from the start (prefix) and full-sequence match
Also checks that GenAI's own chat template + tokenizer give the same prompt ids.

    python genai_parity.py -m <package dir> --provider cuda|cpu|openvino|webgpu [--ov_device CPU|GPU|NPU] \
        --ref hf_ref.json --ref_logits hf_first_logits.npy --out parity.json [--ids p01,p02]
"""

import argparse
import json
import platform
import time

import numpy as np
import onnxruntime_genai as og


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
        for item in args.ep_option:
            key, value = item.split("=", 1)
            config.set_provider_option("OpenVINO", key, value)
    elif args.provider == "cuda":
        config.clear_providers()
        config.append_provider("cuda")
        for item in args.ep_option:
            key, value = item.split("=", 1)
            config.set_provider_option("cuda", key, value)
    elif args.provider == "webgpu":
        import onnxruntime_ep_webgpu as webgpu_ep

        og.register_execution_provider_library(webgpu_ep.get_ep_name(), webgpu_ep.get_library_path())
        config.clear_providers()
        config.append_provider("WebGPU")
        for item in args.ep_option:  # e.g. adapterIndex=1 (on the test PC: 0 = RTX, 1 = Intel GPU)
            key, value = item.split("=", 1)
            config.set_provider_option("WebGPU", key, value)
    elif args.provider == "cpu":
        config.clear_providers()
    if args.overlay:
        config.overlay(args.overlay)
    return og.Model(config)


def last_logits(generator):
    arr = np.asarray(generator.get_logits(), dtype=np.float32)
    return arr.reshape(-1, arr.shape[-1])[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--provider", default="follow_config", choices=["follow_config", "cuda", "cpu", "openvino", "webgpu"])
    parser.add_argument("--ov_device", default="CPU")
    parser.add_argument("--ep_option", action="append", default=[])
    parser.add_argument("--overlay", default="")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ref_logits", default="",
                        help="first-token logits of the reference (.npy from hf_ref.py); without it the first token is "
                             "compared on the reference's top 8 values in --ref")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--skip_forced", action="store_true")
    args = parser.parse_args()

    ref = json.load(open(args.ref, encoding="utf-8"))
    firsts = np.load(args.ref_logits) if args.ref_logits else None
    max_new = ref["meta"]["max_new_tokens"]
    eos = ref["meta"]["eos"]
    eos = set(eos if isinstance(eos, list) else [eos])
    wanted = {i for i in args.ids.split(",") if i}

    t0 = time.perf_counter()
    model = make_model(args)
    tokenizer = og.Tokenizer(model)
    load_s = time.perf_counter() - t0
    out = {"meta": {"model": args.model, "provider": args.provider, "ov_device": args.ov_device if args.provider == "openvino" else "",
                    "ep_option": args.ep_option, "overlay": args.overlay, "og": og.__version__, "device_type": model.device_type,
                    "model_type": model.type, "load_s": load_s, "python": platform.python_version()},
           "records": []}
    for index, record in enumerate(ref["records"]):
        if wanted and record["id"] not in wanted:
            continue
        ids = record["prompt_ids"]
        row = {"id": record["id"], "prompt_len": len(ids)}
        try:
            template = tokenizer.apply_chat_template(json.dumps(record["messages"], ensure_ascii=False), add_generation_prompt=True)
            own = [int(i) for i in tokenizer.encode(template)]
            row["template_ids_equal"] = own == ids
            if own != ids:
                row["template_ids_len"] = len(own)
        except Exception as error:  # the template check is a side note; keep going
            row["template_error"] = str(error)[:300]

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + max_new + 8, do_sample=False)
        try:
            g = og.Generator(model, params)
            t = time.perf_counter()
            g.append_tokens(ids)
            row["prefill_ms"] = (time.perf_counter() - t) * 1000
            first = last_logits(g)
            row["first_top1"] = int(np.argmax(first))
            if firsts is not None:
                ref_first = firsts[index]
                row["first_top1_ref"] = int(np.argmax(ref_first))
                row["first_max_abs"] = float(np.max(np.abs(first - ref_first)))
            else:
                top0 = record["steps"][0]
                row["first_top1_ref"] = int(top0["top_ids"][0])
                row["first_top8_max_abs"] = float(np.max(np.abs(first[top0["top_ids"]] - np.asarray(top0["top_vals"], dtype=np.float32))))
            if not args.skip_forced:
                matches, worst = 1 if row["first_top1"] == row["first_top1_ref"] else 0, 0.0
                steps = record["steps"]
                for k in range(1, len(steps)):
                    g.append_tokens([record["gen_ids"][k - 1]])
                    lk = last_logits(g)
                    top = steps[k]["top_ids"]
                    if int(np.argmax(lk)) == top[0]:
                        matches += 1
                    worst = max(worst, float(np.max(np.abs(lk[top] - np.asarray(steps[k]["top_vals"], dtype=np.float32)))))
                row["forced_steps"] = len(steps)
                row["forced_top1_match"] = matches
                row["forced_top8_max_abs"] = worst
            del g

            g = og.Generator(model, params)
            g.append_tokens(ids)
            gen = []
            t = time.perf_counter()
            while len(gen) < max_new and not g.is_done():
                g.generate_next_token()
                token = int(g.get_next_tokens()[0])
                gen.append(token)
                if token in eos:
                    break
            row["greedy_ms"] = (time.perf_counter() - t) * 1000
            del g
            refgen = record["gen_ids"]
            prefix = 0
            for a, b in zip(gen, refgen):
                if a != b:
                    break
                prefix += 1
            row["greedy_len"] = len(gen)
            row["ref_len"] = len(refgen)
            row["greedy_prefix"] = prefix
            row["greedy_equal"] = gen == refgen
            row["text"] = tokenizer.decode(gen)
        except Exception as error:
            row["error"] = str(error)[:1500]
        out["records"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)

    rows = [r for r in out["records"] if "error" not in r]
    if rows:
        summary = {
            "n": len(rows), "errors": len(out["records"]) - len(rows),
            "first_top1": sum(r["first_top1"] == r["first_top1_ref"] for r in rows),
            "first_max_abs": max((r["first_max_abs"] for r in rows if "first_max_abs" in r), default=None),
            "first_mean_max_abs": float(np.mean([r["first_max_abs"] for r in rows])) if all("first_max_abs" in r for r in rows) else None,
            "first_top8_max_abs": max((r["first_top8_max_abs"] for r in rows if "first_top8_max_abs" in r), default=None),
            "greedy_equal": sum(r["greedy_equal"] for r in rows),
            "greedy_min_prefix": min(r["greedy_prefix"] for r in rows),
            "template_ids_equal": sum(bool(r.get("template_ids_equal")) for r in rows),
        }
        if not args.skip_forced:
            summary["forced_top1"] = f"{sum(r['forced_top1_match'] for r in rows)}/{sum(r['forced_steps'] for r in rows)}"
            summary["forced_top8_max_abs"] = max(r["forced_top8_max_abs"] for r in rows)
        out["summary"] = summary
        print("SUMMARY", json.dumps(summary), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if uses_webgpu(args):
        # The WebGPU plugin 0.5.0 on the RTX adapter kept the process alive after main() returned (outputs written, no
        # CPU use, 2026-11-06): leave without the interpreter's teardown once everything is on disk.
        import os
        import sys

        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
