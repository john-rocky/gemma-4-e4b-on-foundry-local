"""Write the int8 upcast list of the olive-recipes "mixed" decoder recipe for a Gemma 4 E4B export.

The E2B recipe (google-gemma-4-E2B-it/cuda/mixed/text.json) keeps OnnxKQuantQuantization at 4 bits and upcasts to 8 bits,
by exact node name, the MatMuls of mlp down/gate/up_proj and self_attn o_proj in every layer plus lm_head (141 names for
35 layers). Node names depend on the export, so this script reads them from the E4B decoder and writes the same kind of list.

    python mixed_config.py <decoder/model.onnx> <out.json> [--prefix decoder/]
"""

import json
import re
import sys

import onnx

PATTERN = re.compile(r"(mlp/(down|gate|up)_proj|self_attn/o_proj|lm_head)")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
    model = onnx.load(src, load_external_data=False)
    names = [n.name for n in model.graph.node if n.op_type == "MatMul" and PATTERN.search(n.name)]
    config = {prefix + name: {"bits": 8} for name in names}
    kinds = {}
    for name in names:
        kind = PATTERN.search(name).group(1)
        kinds[kind] = kinds.get(kind, 0) + 1
    json.dump(config, open(dst, "w", encoding="utf-8"), indent=1)
    print(len(names), "nodes", kinds, "first", names[:2])


if __name__ == "__main__":
    main()
