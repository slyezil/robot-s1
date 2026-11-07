"""Where does the GPU start to beat one CPU core for a single decision?

Builds decision nets of increasing size (random weights, latency only),
exports each to ONNX, and times batch-1 inference: CPU onnxruntime vs CUDA
graph. Writes to runs/sweep/.

    python sweep_edge.py
"""

import time
from pathlib import Path

import numpy as np
import torch

from export import load
from robot_s1.edge import EdgeDecider, encode_one
from robot_s1.model import DecisionNet, ProbsWrapper
from robot_s1.scenarios import SCENARIOS
from robot_s1.schema import HEADS, N_FEATURES

SIZES = [(256, 3), (512, 4), (1024, 4), (2048, 4)]  # larger: ONNX export runs out of host memory


def bench(fn, x, n=2000):
    for _ in range(100):
        fn(x)
    us = np.empty(n)
    for i in range(n):
        t = time.perf_counter_ns()
        fn(x)
        us[i] = (time.perf_counter_ns() - t) / 1e3
    return np.percentile(us, 50), np.percentile(us, 99)


def main():
    import onnxruntime as ort

    out = Path("runs/sweep")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    x = encode_one(SCENARIOS["cruising"], np.empty(N_FEATURES, np.float32))
    names = [f"{k}_probs" for k in HEADS]
    print(f"{'hidden x layers':16s} {'params':>8s} {'MiB':>5s}   {'cpu ort p50/p99':>18s}   {'cuda graph p50/p99':>20s}   winner")
    for hidden, layers in SIZES:
        torch.manual_seed(0)
        net = DecisionNet(hidden, layers)
        n_params = sum(p.numel() for p in net.parameters())
        ckpt = out / f"h{hidden}_l{layers}.pt"
        torch.save({"state_dict": net.state_dict(), "hidden": hidden, "layers": layers}, ckpt)
        onnx = ckpt.with_suffix(".onnx")
        torch.onnx.export(ProbsWrapper(load(ckpt)), (torch.zeros(1, N_FEATURES),), onnx,
                          input_names=["features"], output_names=names, opset_version=17, dynamo=False)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        buf = np.zeros((1, N_FEATURES), np.float32)

        def run_ort(v):
            buf[0] = v
            return sess.run(None, {"features": buf})

        cpu = bench(run_ort, x)
        d = EdgeDecider(ckpt, "cuda")
        gpu = bench(d.decide_features, x)
        win = "cuda" if gpu[0] < cpu[0] else "cpu"
        print(f"{hidden:5d} x {layers:<8d} {n_params:8,d} {n_params * 4 / 2**20:5.1f}   "
              f"{cpu[0]:8.0f} / {cpu[1]:6.0f} us   {gpu[0]:8.0f} / {gpu[1]:6.0f} us     {win}  ({cpu[0] / gpu[0]:.1f}x)")
        del d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
