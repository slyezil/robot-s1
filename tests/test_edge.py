"""The edge decider must give the same answers as the plain model."""

import numpy as np
import pytest
import torch

from export import load
from robot_s1.edge import EdgeDecider, encode_one
from robot_s1.scenarios import SCENARIOS
from robot_s1.schema import BEHAVIORS, HEADS, N_FEATURES
from robot_s1.states import encode, from_json

CKPT = "runs/latest/model.pt"


def test_encode_one_matches_batch_encoder():
    objs = list(SCENARIOS.values())
    single = np.stack([encode_one(o, np.empty(N_FEATURES, np.float32)) for o in objs])
    batch = encode(from_json(objs)).numpy()
    assert np.allclose(single, batch, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_edge_matches_model(device):
    net = load(CKPT)
    d = EdgeDecider(CKPT, device)
    for name, s in SCENARIOS.items():
        got = d.decide(s)
        with torch.no_grad():
            out = net(encode(from_json([s])))
        want = {k: out[k].softmax(-1)[0].numpy() for k in HEADS}
        for k in HEADS:
            assert np.allclose(got["probs"][k], want[k], atol=1e-4), (name, k)
        assert got["behavior"] == BEHAVIORS[int(want["behavior"].argmax())]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_graph_replay_sees_new_inputs():
    # A captured graph reading a stale buffer would return the same answer for every state.
    d = EdgeDecider(CKPT, "cuda")
    a = d.decide(SCENARIOS["cruising"])["behavior"]
    b = d.decide(SCENARIOS["payload slipped"])["behavior"]
    assert (a, b) == ("continue_task", "request_human_help")
