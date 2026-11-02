"""Vectorized port of goSysOne's hand-weighted Heuristic (robot/heuristic.go).

This is the first teacher: the small model learns to reproduce its full
probability distributions. Keep it in lockstep with the Go code; later
teachers (recorded runs, hand-labelled edge cases) can be mixed in on top.
"""

import torch
import torch.nn.functional as F

from .states import State


def hazard_level(s: State) -> torch.Tensor:
    """Combines factors into [0, 1]. Mirrors hazardLevel()."""
    h = s["human_nearby"] * (
        0.25 + 0.35 * F.relu(1 - s["human_distance_m"] / 8) + 0.1 * s["human_approaching"]
    )
    h = h + 0.4 * F.relu(1 - s["nearest_obstacle_m"] / 4)
    h = h + 0.4 * (1 - s["payload_secure"])
    h = h + 0.2 * (s["motor_temp_c"] > 70).float()
    return h.clamp(max=1)


def behavior_logits(s: State, hazard: torch.Tensor) -> torch.Tensor:
    """Mirrors behaviorLogits(); column order matches schema.BEHAVIORS."""
    low_battery = F.relu((25 - s["battery_pct"]) / 25)
    pressure = low_battery * (s["distance_to_goal_m"] / 8).clamp(max=1)
    unsecure = 1 - s["payload_secure"]
    return torch.stack(
        [
            2 - 5 * hazard - 3 * pressure,
            0.5 + 3 * hazard - 4 * F.relu(hazard - 0.7),
            -1 + 6 * F.relu(hazard - 0.5),
            -2 + 9 * pressure,
            -3 + 6 * unsecure + s["failed_grasps"],
        ],
        dim=1,
    )


def hazard_logits(hazard: torch.Tensor, levels: int = 5) -> torch.Tensor:
    i = torch.arange(levels, device=hazard.device, dtype=hazard.dtype)
    return -2 * (i[None, :] - hazard[:, None] * (levels - 1)).abs()


def needs_human_logits(s: State) -> torch.Tensor:
    p = 3 * (1 - s["payload_secure"]) + s["failed_grasps"]
    return torch.stack([torch.full_like(p, 1.5), p], dim=1)


def teacher_probs(s: State) -> dict[str, torch.Tensor]:
    h = hazard_level(s)
    return {
        "behavior": behavior_logits(s, h).softmax(-1),
        "hazard": hazard_logits(h).softmax(-1),
        "needs_human": needs_human_logits(s).softmax(-1),
    }
