"""The teacher must match goSysOne's robot/heuristic.go. Expected values are
worked by hand from the Go formulas."""

import math

import pytest
import torch

from robot_s1.scenarios import SCENARIOS
from robot_s1.schema import BEHAVIORS, N_FEATURES
from robot_s1.states import encode, from_json, sample_states
from robot_s1.teacher import hazard_level, teacher_probs


def probs(name):
    return teacher_probs(from_json([SCENARIOS[name]]))


def top(name):
    return BEHAVIORS[probs(name)["behavior"][0].argmax()]


@pytest.mark.parametrize(
    "name, want",
    [
        ("cruising", "continue_task"),
        ("payload slipped", "request_human_help"),
        ("low battery, 3m from goal", "continue_task"),  # deliver first
        ("low battery, 35m from goal", "return_to_dock"),
        ("pallet 1m ahead", "slow_down"),
        ("person 2m approaching", "slow_down"),
    ],
)
def test_behavior(name, want):
    assert top(name) == want


def test_hazard_person_2m():
    # 0.25 + 0.35*(1-2/8) + 0.1 (approaching) + 0.4*(1-2/4) (obstacle = person)
    h = hazard_level(from_json([SCENARIOS["person 2m approaching"]]))
    assert h.item() == pytest.approx(0.8125, abs=1e-6)


def test_needs_human_payload_slipped():
    # logits [1.5, 3] -> sigmoid(1.5)
    p = probs("payload slipped")["needs_human"][0, 1].item()
    assert p == pytest.approx(1 / (1 + math.exp(-1.5)), abs=1e-6)


def test_behavior_logits_payload_slipped():
    # continue 0, slow 1.7, stop -1, dock -2, help 3 (hazard 0.4, no battery pressure)
    want = torch.tensor([0.0, 1.7, -1.0, -2.0, 3.0]).softmax(-1)
    assert torch.allclose(probs("payload slipped")["behavior"][0], want, atol=1e-6)


def test_sampler_and_encoding_shapes():
    s = sample_states(1000, torch.Generator().manual_seed(0), torch.device("cpu"))
    x = encode(s)
    assert x.shape == (1000, N_FEATURES)
    assert x.min() >= 0 and x.max() <= 1
    for v in teacher_probs(s).values():
        assert torch.allclose(v.sum(-1), torch.ones(1000), atol=1e-5)
