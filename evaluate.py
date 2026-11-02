"""Compare student vs teacher on the named demo scenarios and measure latency.

    python evaluate.py
"""

import argparse
import time
from pathlib import Path

import torch

from export import load
from robot_s1.scenarios import SCENARIOS
from robot_s1.schema import BEHAVIORS, HEADS
from robot_s1.states import encode, from_json, sample_states
from robot_s1.teacher import teacher_probs


def bench(fn, iters: int, sync=lambda: None) -> float:
    for _ in range(10):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/latest/model.pt"))
    args = ap.parse_args()
    net = load(args.ckpt)

    names = list(SCENARIOS)
    s = from_json(list(SCENARIOS.values()))
    t = teacher_probs(s)
    with torch.no_grad():
        p = {k: v.softmax(-1) for k, v in net(encode(s)).items()}

    print(f"{'scenario':28s} {'teacher':20s} {'student':20s} p_student  hazard(t/s)  p_human(t/s)")
    lv = torch.arange(len(HEADS["hazard"]), dtype=torch.float32) / (len(HEADS["hazard"]) - 1)
    agree = 0
    for i, n in enumerate(names):
        tb, sb = BEHAVIORS[t["behavior"][i].argmax()], BEHAVIORS[p["behavior"][i].argmax()]
        agree += tb == sb
        print(
            f"{n:28s} {tb:20s} {sb:20s} {p['behavior'][i].max():.2f}       "
            f"{(t['hazard'][i] * lv).sum():.2f}/{(p['hazard'][i] * lv).sum():.2f}    "
            f"{t['needs_human'][i][1]:.2f}/{p['needs_human'][i][1]:.2f}"
            + ("" if tb == sb else "   <-- disagree")
        )
    print(f"behavior agreement on scenarios: {agree}/{len(names)}")

    print("\nlatency (forward pass, all 3 questions):")
    x1 = encode(from_json([SCENARIOS["cruising"]]))
    with torch.no_grad():
        print(f"  cpu  batch 1      {bench(lambda: net(x1), 2000) * 1e6:8.1f} us")
        if torch.cuda.is_available():
            g = net.cuda()
            sync = torch.cuda.synchronize
            xg1 = x1.cuda()
            xb = encode(sample_states(65536, torch.Generator(device="cuda").manual_seed(0), torch.device("cuda")))
            print(f"  cuda batch 1      {bench(lambda: g(xg1), 2000, sync) * 1e6:8.1f} us")
            dt = bench(lambda: g(xb), 50, sync)
            print(f"  cuda batch 65536  {dt * 1e3:8.2f} ms  ({65536 / dt / 1e6:.1f}M decisions/s)")


if __name__ == "__main__":
    main()
