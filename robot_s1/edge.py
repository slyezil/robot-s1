"""Edge inference: one robot, one decision at a time, lowest latency.

Training cares about states per second; a robot cares about how long one
decision takes, every time. For a 140k-parameter model the math is nothing;
the cost is launching kernels and moving bytes. So the EdgeDecider:

- allocates its input/output buffers once (pinned host memory, static device
  tensors), so no allocation happens per decision;
- captures the whole forward pass (encode-free: the caller passes the feature
  vector) plus softmax into one CUDA graph and replays it, which turns ~15
  kernel launches into one;
- copies 18 floats in and 12 floats out, nothing else.

Use `EdgeDecider(path).decide(state_dict)` or, when the features are already
built, `.decide_features(np.ndarray)`.
"""

from pathlib import Path

import numpy as np
import torch

from .model import DecisionNet
from .schema import BEHAVIORS, FEATURES, HEADS, N_FEATURES

N_OUT = sum(len(v) for v in HEADS.values())


def encode_one(s: dict, out: np.ndarray) -> np.ndarray:
    """One robot.State JSON object -> feature vector, written into `out`."""
    last = s.get("last_behavior", "continue_task")
    for i, f in enumerate(FEATURES):
        if f.kind == "num":
            v = float(s.get(f.name, 99.0 if f.name == "nearest_obstacle_m" else 0.0))
            out[i] = (min(max(v, f.lo), f.hi) - f.lo) / (f.hi - f.lo)
        elif f.kind == "bool":
            out[i] = 1.0 if s.get(f.name, False) else 0.0
        else:
            out[i] = 1.0 if last == f.value else 0.0
    return out


class _Flat(torch.nn.Module):
    """Model -> one concatenated probability vector, so one output copy suffices."""

    def __init__(self, net: DecisionNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        o = self.net(x)
        return torch.cat([o[k].softmax(-1) for k in HEADS], dim=-1)


class EdgeDecider:
    def __init__(self, ckpt: str | Path, device: str = "cuda", fp16: bool = False, graph: bool = True):
        c = torch.load(ckpt, map_location="cpu")
        net = DecisionNet(c["hidden"], c["layers"])
        net.load_state_dict(c["state_dict"])
        self.dev = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = _Flat(net).to(self.dev, self.dtype).eval()
        self.use_graph = graph and self.dev.type == "cuda"

        # Static buffers. Pinned host memory lets the copy go straight to the GPU.
        pin = self.dev.type == "cuda"
        self.h_in = torch.zeros(1, N_FEATURES, dtype=torch.float32, pin_memory=pin)
        self.h_out = torch.zeros(1, N_OUT, dtype=torch.float32, pin_memory=pin)
        self.d_in = torch.zeros(1, N_FEATURES, dtype=self.dtype, device=self.dev)
        self.d_out = torch.zeros(1, N_OUT, dtype=torch.float32, device=self.dev)
        self.np_in = self.h_in.numpy()[0]
        self.np_out = self.h_out.numpy()[0]

        if self.use_graph:
            self.stream = torch.cuda.Stream()
            # Warm up on a side stream, then capture.
            with torch.no_grad(), torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.d_out.copy_(self.model(self.d_in))
            torch.cuda.current_stream().wait_stream(self.stream)
            self.g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self.g, stream=self.stream):
                self.d_out.copy_(self.model(self.d_in))
        self._split = np.cumsum([len(v) for v in HEADS.values()])[:-1]

    @torch.no_grad()
    def decide_features(self, feats: np.ndarray) -> np.ndarray:
        """Feature vector [N_FEATURES] -> concatenated probs [N_OUT] (a view of a reused buffer)."""
        self.np_in[:] = feats
        if self.dev.type == "cuda":
            with torch.cuda.stream(self.stream if self.use_graph else torch.cuda.current_stream()):
                self.d_in.copy_(self.h_in, non_blocking=True)
                if self.use_graph:
                    self.g.replay()
                else:
                    self.d_out.copy_(self.model(self.d_in))
                self.h_out.copy_(self.d_out, non_blocking=True)
            (self.stream if self.use_graph else torch.cuda.current_stream()).synchronize()
        else:
            self.h_out.copy_(self.model(self.h_in))
        return self.np_out

    def decide(self, state: dict) -> dict:
        """robot.State JSON object -> {'behavior': name, 'probs': {...}, 'hazard': expected, 'needs_human': p}."""
        p = self.decide_features(encode_one(state, np.empty(N_FEATURES, np.float32)))
        b, h, nh = np.split(p, self._split)
        return {
            "behavior": BEHAVIORS[int(b.argmax())],
            "probs": {k: v.tolist() for k, v in zip(HEADS, (b, h, nh))},
            "hazard": float((h * np.arange(len(h)) / (len(h) - 1)).sum()),
            "needs_human": float(nh[1]),
        }

    def gpu_memory_mib(self) -> float:
        return torch.cuda.memory_allocated(self.dev) / 2**20 if self.dev.type == "cuda" else 0.0
