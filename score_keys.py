"""Score fixture answers against the pre-registered keys (fixture/answer_keys.json) and the Hugging Face reference.

    python score_keys.py --keys fixture/answer_keys.json --ref hf_ref.json <result.json> [<result.json> ...]

A result file holds {"records": [{"id": ..., "text": ...}, ...]} (hf_ref.py, genai_parity.py, fl_rest.py fixture).
Bar: the form satisfies the key on every prompt where the reference does, with at most one miss.
"""

import argparse
import json
import re
import sys


def passes(key, text):
    t = (text or "").lower()
    if "min_lines" in key:
        return len([line for line in (text or "").splitlines() if line.strip()]) >= key["min_lines"]
    def has(s):
        s = s.lower()
        if key.get("word"):
            return re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", t) is not None
        return s in t
    if "all" in key:
        return all(has(s) for s in key["all"])
    return any(has(s) for s in key["any"])


def score(keys, ref_ok, path):
    data = json.load(open(path, encoding="utf-8"))
    rows = {r["id"]: r for r in data["records"]}
    ok, miss, errors = [], [], []
    for pid, key in keys.items():
        r = rows.get(pid)
        if r is None or "error" in r:
            errors.append(pid)
            continue
        (ok if passes(key, r.get("text")) else miss).append(pid)
    counted = [p for p in ref_ok if p in ok or p in miss or p in errors]
    misses_on_ref_ok = [p for p in ref_ok if p not in ok]
    verdict = "PASS" if len(misses_on_ref_ok) <= 1 else "FAIL"
    return {"file": path, "keys_ok": len(ok), "of": len(keys), "misses": miss, "errors": errors,
            "ref_ok": len(ref_ok), "misses_where_ref_ok": misses_on_ref_ok, "bar": verdict, "counted": len(counted)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    keys = json.load(open(args.keys, encoding="utf-8"))["keys"]
    ref = json.load(open(args.ref, encoding="utf-8"))
    ref_ok = [r["id"] for r in ref["records"] if r["id"] in keys and passes(keys[r["id"]], r["text"])]
    print(json.dumps({"reference": args.ref, "ref_keys_ok": len(ref_ok), "ref_misses": [p for p in keys if p not in ref_ok]}))
    for path in args.results:
        print(json.dumps(score(keys, ref_ok, path), ensure_ascii=False))


if __name__ == "__main__":
    # Output redirected to a file on Windows is written in the ANSI code page (cp932 on a Japanese system), which
    # cannot encode every character of a model's answer: write UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
