"""Measure what a robot would feel: the latency of one decision, repeated.

    python bench_edge.py                    # all variants, 5000 decisions each
    python bench_edge.py --ckpt runs/improved/model.pt --n 20000

Reports p50/p99/max per decision and how many would miss a deadline. The
closed-loop section drives the simulator one robot at a time, with the
decider in the loop, so the input changes every call like real sensors.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from robot_s1.edge import EdgeDecider, N_OUT, encode_one
from robot_s1.scenarios import SCENARIOS
from robot_s1.schema import N_FEATURES
from robot_s1.sim import Env, SimConfig


def percentiles(us: np.ndarray) -> str:
    p = np.percentile(us, [50, 90, 99, 99.9])
    return f"p50 {p[0]:7.1f}  p90 {p[1]:7.1f}  p99 {p[2]:7.1f}  p99.9 {p[3]:7.1f}  max {us.max():7.1f} us"


def bench(fn, inputs: np.ndarray, n: int) -> np.ndarray:
    for i in range(200):
        fn(inputs[i % len(inputs)])
    us = np.empty(n)
    for i in range(n):
        x = inputs[i % len(inputs)]
        t = time.perf_counter_ns()
        fn(x)
        us[i] = (time.perf_counter_ns() - t) / 1e3
    return us


def onnx_runner(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    buf = np.zeros((1, N_FEATURES), np.float32)

    def run(x):
        buf[0] = x
        return np.concatenate(sess.run(None, {"features": buf}), axis=1)[0]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/improved/model.pt"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--deadline-us", type=float, default=1000.0, help="goSysOne's controller runs at 1 kHz")
    args = ap.parse_args()
    torch.set_num_threads(1)  # a robot would pin this; also avoids thread wake-up jitter

    inputs = np.stack([encode_one(s, np.empty(N_FEATURES, np.float32)) for s in SCENARIOS.values()])
    variants = [
        ("cpu torch (1 thread)", lambda: EdgeDecider(args.ckpt, "cpu")),
    ]
    onnx = args.ckpt.with_name("robot_s1.onnx")
    if onnx.exists():
        variants.append(("cpu onnxruntime (1 thread)", lambda: onnx_runner(onnx)))
    if torch.cuda.is_available():
        variants += [
            ("cuda eager fp32", lambda: EdgeDecider(args.ckpt, "cuda", graph=False)),
            ("cuda graph fp32", lambda: EdgeDecider(args.ckpt, "cuda")),
            ("cuda graph fp16", lambda: EdgeDecider(args.ckpt, "cuda", fp16=True)),
        ]

    ref = None
    print(f"one decision (all 3 questions), {args.n} calls each, deadline {args.deadline_us:.0f} us:")
    for name, make in variants:
        d = make()
        fn = d.decide_features if isinstance(d, EdgeDecider) else d
        out = np.array(fn(inputs[0]), copy=True)
        if ref is None:
            ref = out
        err = float(np.abs(out - ref).max())
        us = bench(fn, inputs, args.n)
        miss = 100 * (us > args.deadline_us).mean()
        mem = f"  gpu {d.gpu_memory_mib():.1f} MiB" if isinstance(d, EdgeDecider) and d.dev.type == "cuda" else ""
        print(f"  {name:28s} {percentiles(us)}  miss {miss:5.2f}%  |probs err| {err:.1e}{mem}")

    if not torch.cuda.is_available():
        return
    print("\nclosed loop: GPU decider drives one simulated robot, decision every physics tick")
    d = EdgeDecider(args.ckpt, "cuda")
    cfg = SimConfig(decide_every=1)
    env = Env.random(1, cfg, torch.Generator().manual_seed(1), "cpu")
    gen = torch.Generator().manual_seed(2)
    feats = np.empty(N_FEATURES, np.float32)
    lat, steps = [], 0
    t_wall = time.perf_counter()
    while not bool(env.done.all()) and steps < 4000:
        s = {k: v.item() for k, v in env.sense().items()}
        s["last_behavior"] = ["continue_task", "slow_down", "stop_and_wait", "return_to_dock", "request_human_help"][int(s["last_behavior"])]
        t = time.perf_counter_ns()
        p = d.decide_features(encode_one(s, feats))
        lat.append((time.perf_counter_ns() - t) / 1e3)
        env.decide(torch.tensor([int(p[:5].argmax())]), gen, max_time=cfg.max_time)
        steps += 1
    wall = time.perf_counter() - t_wall
    us = np.array(lat)
    outcome = [k for k in ("delivered", "stranded", "dropped", "fault", "timeout") if env.f[k].item()]
    print(f"  {steps} decisions, {outcome[0] if outcome else 'running'} at t={env.t.item():.1f}s sim time")
    print(f"  decision latency: {percentiles(us)}  miss {100 * (us > args.deadline_us).mean():.2f}%")
    print(f"  whole loop incl. sim + encoding: {1e6 * wall / steps:.0f} us/step ({steps / wall:.0f} Hz)")


if __name__ == "__main__":
    main()
