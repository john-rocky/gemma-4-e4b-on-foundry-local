"""Run an Olive config in this process, with the k-quant search of OnnxKQuantQuantization done in chunks of groups.

Olive 0.13.0 quantizes each weight matrix in one numpy call that holds about eight float32 copies of the matrix and a
float64 result at once. For the Gemma 4 E4B lm_head (2560 x 262144 weights) that count comes to about 27 GB, more than
a 32 GB machine has free next to the loaded decoder. The k-quant search is independent for every group of block_size
values, so running it over chunks of groups gives the same output with a peak of a few GB.

    python olive_run_chunked.py <olive config.json>
"""

import sys

import numpy as np
import olive.passes.onnx.kquant_quantization as kq
from olive.workflows import run

GROUPS_PER_CHUNK = 1 << 22  # 4M groups of 32 values: about 0.5 GB per float32 copy


def kquant_quantize_chunked(data, num_bits=4, group_size=32):
    flat = np.reshape(data, (-1, group_size))
    n = flat.shape[0]
    # The caller (_kquant_block_quant) casts the quantized values to uint8 right away; they are integers in [0, 2**bits - 1].
    q = np.empty((n, group_size), dtype=np.uint8)
    scale = np.empty((n, 1), dtype=np.float64)
    zero_point = np.empty((n, 1), dtype=np.uint8)
    for start in range(0, n, GROUPS_PER_CHUNK):
        end = min(n, start + GROUPS_PER_CHUNK)
        q[start:end], scale[start:end], zero_point[start:end] = kq._kquant_quantize_cpu(flat[start:end], num_bits, group_size)
    return q, scale, zero_point


kq._kquant_quantize = kquant_quantize_chunked

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run(sys.argv[1])
