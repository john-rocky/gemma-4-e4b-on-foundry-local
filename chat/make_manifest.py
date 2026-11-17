"""Write models.json: the files of the model packages chat.ps1 downloads, pinned to one revision of the Hugging Face repo,
with their sizes and SHA-256 hashes. Large files take the hash from the Hub's file listing (the LFS object id is the
file's SHA-256); small files are downloaded here and hashed, and their git blob id is checked against the listing.

    python make_manifest.py [--revision <commit>] [--out models.json]

Standard library only. The chat itself never runs this: it reads the models.json committed next to it.
"""
import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "mlboydaisuke/gemma-4-E4B-it-Foundry-Local"
REVISION = "ad8a408b845aed1abe40806bb2b734a189817054"
PACKAGES = ["cuda-int4-8k", "cpu-int4"]
DEVICES = {"cuda": "cuda-int4-8k", "webgpu": "cuda-int4-8k", "cpu": "cpu-int4"}
API = "https://huggingface.co/api/models/{repo}/tree/{revision}/{folder}?recursive=1"
URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "make_manifest.py"}),
                                timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", default=os.path.join(HERE, "models.json"))
    args = ap.parse_args()
    out = {"repo": REPO, "revision": args.revision,
           "url": URL.format(repo=REPO, revision=args.revision, path="{package}/{path}"),
           "devices": DEVICES, "packages": {}}
    for package in PACKAGES:
        listing = json.loads(get(API.format(repo=REPO, revision=args.revision, folder=package)))
        files = []
        for item in sorted((i for i in listing if i["type"] == "file"), key=lambda i: i["path"]):
            path = item["path"][len(package) + 1:]
            if "lfs" in item:
                sha256 = item["lfs"]["oid"]
            else:
                data = get(URL.format(repo=REPO, revision=args.revision,
                                      path=urllib.parse.quote(item["path"])))
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                if len(data) != item["size"] or blob != item["oid"]:
                    raise SystemExit(f"{item['path']}: {len(data)} bytes, git blob {blob}; the listing says "
                                     f"{item['size']} bytes, {item['oid']}")
                sha256 = hashlib.sha256(data).hexdigest()
            files.append({"path": path, "size": item["size"], "sha256": sha256})
        out["packages"][package] = {"bytes": sum(f["size"] for f in files), "files": files}
        print(f"{package}: {len(files)} files, {out['packages'][package]['bytes']:,} bytes")
    with open(args.out + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    os.replace(args.out + ".tmp", args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
