"""Export a trained checkpoint to ONNX and verify it against PyTorch.

    python export.py                    # runs/latest/model.pt -> runs/latest/robot_s1.onnx + features.json

Inputs:  features float32 [N, F]   (encoding described in features.json)
Outputs: behavior_probs [N, 5], hazard_probs [N, 5], needs_human_probs [N, 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from robot_s1.model import DecisionNet, ProbsWrapper
from robot_s1.schema import HEADS, N_FEATURES, feature_spec
from robot_s1.states import encode, sample_states


def load(ckpt: Path) -> DecisionNet:
    c = torch.load(ckpt, map_location="cpu")
    net = DecisionNet(c["hidden"], c["layers"])
    net.load_state_dict(c["state_dict"])
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/latest/model.pt"))
    args = ap.parse_args()

    net = load(args.ckpt)
    wrapper = ProbsWrapper(net).eval()
    out_dir = args.ckpt.parent
    onnx_path = out_dir / "robot_s1.onnx"
    names = [f"{k}_probs" for k in HEADS]

    torch.onnx.export(
        wrapper,
        (torch.zeros(1, N_FEATURES),),
        onnx_path,
        input_names=["features"],
        output_names=names,
        dynamic_axes={"features": {0: "n"}, **{n: {0: "n"} for n in names}},
        opset_version=17,
        dynamo=False,
    )
    (out_dir / "features.json").write_text(json.dumps(feature_spec(), indent=2))
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KiB) and features.json")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(123)
    x = encode(sample_states(4096, gen, torch.device("cpu")))
    with torch.no_grad():
        ref = wrapper(x)
    got = sess.run(names, {"features": x.numpy()})
    err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    print(f"onnxruntime vs torch max abs diff: {err:.2e}")
    assert err < 1e-4, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    main()
