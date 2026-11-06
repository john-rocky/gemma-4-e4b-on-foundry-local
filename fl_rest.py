"""Gemma 4 E4B through the running Foundry Local daemon (OpenAI-compatible REST at the daemon's web URL).

fixture: the 24 prompts of hf_ref.json, temperature 0, the reference's max new tokens; compares the text with the reference's.
speed:   one long prompt (the same text as genai_speed.py + a request to continue), streamed; time to first token and
         tokens per second after it (one SSE chunk = one token as Foundry streams), W warm-ups then R timed runs.

    python fl_rest.py fixture <base url>/v1 <model id> hf_ref.json out.json
    python fl_rest.py speed   <base url>/v1 <model id> out.jsonl [runs]
"""

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

TEXT = ("The lighthouse keeper climbed the spiral stairs every evening, counting the steps as his father had taught him. "
        "From the lamp room he could see the harbor, the fishing boats returning with their nets, and the long line of the coast "
        "fading into the fog. He wrote the weather in a worn notebook: wind from the west, waves moderate, visibility falling. ")


def post(url, body, timeout=900):
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)


def fixture(url, model, ref_path, out_path):
    ref = json.load(open(ref_path, encoding="utf-8"))
    max_new = ref["meta"]["max_new_tokens"]
    rows = []
    for record in ref["records"]:
        row = {"id": record["id"], "ref_prompt_tokens": len(record["prompt_ids"]), "ref_new_tokens": len(record["gen_ids"])}
        t = time.perf_counter()
        try:
            with post(url, {"model": model, "messages": record["messages"], "max_tokens": max_new, "temperature": 0}) as response:
                data = json.loads(response.read().decode("utf-8"))
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            text = choice["message"]["content"] or ""
            row.update({"text": text, "finish": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"), "s": time.perf_counter() - t})
            want = record["text"]
            row["equal"] = text.strip() == want.strip()
            same = 0
            for a, b in zip(text.strip(), want.strip()):
                if a != b:
                    break
                same += 1
            row["common_prefix_chars"] = same
            row["ref_chars"] = len(want.strip())
        except urllib.error.HTTPError as error:
            row["error"] = f"HTTP {error.code} {error.read().decode('utf-8', 'replace')[:1500]}"
        except Exception as error:
            row["error"] = repr(error)[:1500]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    good = [r for r in rows if "error" not in r]
    summary = {"n": len(rows), "errors": len(rows) - len(good), "equal": sum(r["equal"] for r in good),
               "prompt_tokens_equal_ref": sum(r.get("prompt_tokens") == r["ref_prompt_tokens"] for r in good)}
    print("SUMMARY", json.dumps(summary), flush=True)
    json.dump({"model": model, "url": url, "summary": summary, "records": rows}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if summary["errors"]:
        sys.exit(f"{summary['errors']} of {summary['n']} prompts got no answer (first error: {next(r['error'] for r in rows if 'error' in r)[:300]})")


def speed(url, model, out_path, runs=3, warmup=1):
    prompt = TEXT * 13 + "\nContinue this story for at least 400 words."
    rows = []
    for run in range(warmup + runs):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 160, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        t = time.perf_counter()
        first = last = None
        chunks = 0
        usage = {}
        with post(url, body) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
        row = {"run": run, "warmup": run < warmup, "ttft_ms": (first - t) * 1000 if first else None, "chunks": chunks,
               "decode_tok_s": (chunks - 1) / (last - first) if first and last and chunks > 1 else None,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        rows.append(row)
        print(json.dumps(row), flush=True)
    timed = [r for r in rows if not r["warmup"] and r["ttft_ms"]]
    summary = {"summary": True, "model": model, "runs": len(timed),
               "ttft_ms_median": statistics.median(r["ttft_ms"] for r in timed) if timed else None,
               "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in timed if r["decode_tok_s"]) if timed else None,
               "prompt_tokens": timed[-1]["prompt_tokens"] if timed else None}
    print("SUMMARY", json.dumps(summary), flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] not in ("fixture", "speed"):
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "fixture":
        fixture(sys.argv[2].rstrip("/"), sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        speed(sys.argv[2].rstrip("/"), sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
