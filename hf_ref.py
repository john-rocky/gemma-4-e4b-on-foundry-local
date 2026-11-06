"""Hugging Face reference for the fixture: google/gemma-4-E4B-it in bfloat16 (the checkpoint's dtype) on the CPU.

For each prompt: the chat-template token ids, greedy decoding of up to N new tokens, the full logits
of the first new token (saved as float32 .npy) and the top-8 logits of every later step.

    python hf_ref.py --model <dir with config.json + model.safetensors> --fixture prompts.json \
        --out hf_ref.json --logits hf_first_logits.npy
"""

import argparse
import json
import platform
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


def prompt_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--max_new", type=int, default=0, help="0 = the fixture's max_new_tokens")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    fixture = json.load(open(args.fixture, encoding="utf-8"))
    max_new = args.max_new or fixture["max_new_tokens"]
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    placement = "all on the CPU"
    if args.device == "cuda":
        # bf16 weights are 16 GB, the RTX has 16 GB: the text decoder goes to the GPU; the per-layer embedding table
        # (2.8B parameters, a lookup) and the unused vision / audio towers stay on the CPU.
        # generate() moves the ids to model.device (the first parameter's device), so everything else goes to the GPU and
        # the table is re-attached last, after the move.
        lm = model.model.language_model
        ple = lm._modules.pop("embed_tokens_per_layer")
        model.to("cuda")
        lm._modules["embed_tokens_per_layer"] = ple
        ple_forward = ple.forward
        ple.forward = lambda input_ids, *a, **k: ple_forward(input_ids.to("cpu"), *a, **k).to("cuda")
        placement = "all on the GPU except embed_tokens_per_layer (2.82B parameters, a lookup) on the CPU"
    load_s = time.perf_counter() - t0
    eos = model.generation_config.eos_token_id
    result = {"meta": {"model": args.model, "dtype": "bfloat16", "device": args.device, "placement": placement, "transformers": transformers.__version__,
                       "torch": torch.__version__, "threads": torch.get_num_threads(), "python": platform.python_version(),
                       "max_new_tokens": max_new, "eos": eos, "load_s": load_s, "fixture": fixture["name"]},
              "records": []}
    firsts = []
    for record in fixture["records"]:
        ids = prompt_ids(tokenizer, record["messages"])
        input_ids = torch.tensor([ids], device=args.device)
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new,
                                 do_sample=False, temperature=None, top_p=None, top_k=None,
                                 output_logits=True, return_dict_in_generate=True)
        ms = (time.perf_counter() - t) * 1000
        gen = out.sequences[0, len(ids):].tolist()
        steps = []
        for i, logits in enumerate(out.logits):
            row = logits[0].float().cpu()
            if i == 0:
                firsts.append(row.numpy().astype(np.float32))
            values, indices = torch.topk(row, 8)
            steps.append({"top_ids": indices.tolist(), "top_vals": [round(v, 5) for v in values.tolist()]})
        text = tokenizer.decode(gen, skip_special_tokens=True)
        result["records"].append({"id": record["id"], "messages": record["messages"], "prompt_ids": ids, "gen_ids": gen,
                                  "text": text, "steps": steps, "ms": ms})
        print(f"{record['id']} prompt={len(ids)} new={len(gen)} {ms / 1000:.1f}s {text[:60]!r}", flush=True)
    np.save(args.logits, np.stack(firsts))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"done {len(result['records'])} records, load {load_s:.1f} s")


if __name__ == "__main__":
    # Output redirected to a file on Windows is written in the ANSI code page (cp932 on a Japanese system), which
    # cannot encode every character of a model's answer: write UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
