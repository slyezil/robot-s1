"""Batched robot states as a dict of tensors, plus a random state sampler and
the feature encoder. Everything runs on whatever device you pass (GPU for
training), so a batch of 100k states costs a few milliseconds.
"""

import json

import torch

from .schema import BEHAVIORS, FEATURES, NO_OBSTACLE_M

State = dict[str, torch.Tensor]  # field name -> [N] tensor (float32; bools as 0/1)


def sample_states(n: int, gen: torch.Generator, device: torch.device) -> State:
    """Random states covering the goSysOne corridor scenarios and well beyond.

    The mix is deliberately wider than the scripted demo so the model sees
    combinations (low battery + person close + hot motor ...) that the demo
    rarely produces.
    """

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)

    def coin(p):
        return (torch.rand(n, generator=gen, device=device) < p).float()

    # Battery: oversample the interesting low range.
    battery = torch.where(coin(0.4).bool(), u(0, 30), u(0, 100))

    corridor = u(10, 60)
    progress = torch.rand(n, generator=gen, device=device)
    dist_goal = corridor * (1 - progress)
    dist_dock = corridor * progress

    # Obstacles: half the time nothing in range (sentinel 99).
    has_obst = coin(0.5)
    obst = torch.where(has_obst.bool(), u(0, 15), torch.full_like(battery, NO_OBSTACLE_M))

    human = coin(0.4)
    human_dist = u(0, 15) * human
    human_approach = coin(0.8) * human
    # Sense() folds a nearby person into nearest_obstacle_m.
    obst = torch.where(human.bool(), torch.minimum(obst, human_dist), obst)

    secure = 1 - coin(0.15)
    # Failed grasps: mostly 0, occasionally 1-4.
    grasps = torch.floor(torch.clamp(-torch.log(torch.rand(n, generator=gen, device=device)) * 0.6, max=4.0))

    last = torch.randint(0, len(BEHAVIORS), (n,), generator=gen, device=device)

    return {
        "battery_pct": battery,
        "speed_mps": u(-1.5, 1.5),
        "distance_to_goal_m": dist_goal,
        "distance_to_dock_m": dist_dock,
        "nearest_obstacle_m": obst,
        "human_nearby": human,
        "human_distance_m": human_dist,
        "human_approaching": human_approach,
        "payload_kg": u(0, 30),
        "payload_secure": secure,
        "failed_grasps": grasps,
        "motor_temp_c": u(25, 95),
        "seconds_on_task": u(0, 600),
        "last_behavior": last.float(),  # index into BEHAVIORS
    }


def encode(s: State) -> torch.Tensor:
    """State -> [N, N_FEATURES] float32, following schema.FEATURES."""
    cols = []
    for f in FEATURES:
        if f.kind == "num":
            x = s[f.name].clamp(f.lo, f.hi)
            cols.append((x - f.lo) / (f.hi - f.lo))
        elif f.kind == "bool":
            cols.append(s[f.name].float())
        elif f.kind == "onehot":
            cols.append((s["last_behavior"] == BEHAVIORS.index(f.value)).float())
        else:
            raise ValueError(f.kind)
    return torch.stack(cols, dim=1)


def from_json(objs: list[dict], device: torch.device | str = "cpu") -> State:
    """goSysOne robot.State JSON objects -> State. Omitted fields default like Go's zero values."""
    defaults = {"nearest_obstacle_m": NO_OBSTACLE_M, "last_behavior": "continue_task"}
    out: dict[str, list[float]] = {}
    for f in [f for f in FEATURES if f.kind != "onehot"]:
        out[f.name] = [float(o.get(f.name, defaults.get(f.name, 0))) for o in objs]
    out["last_behavior"] = [
        float(BEHAVIORS.index(o.get("last_behavior", defaults["last_behavior"]))) for o in objs
    ]
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in out.items()}


def from_json_str(s: str, device="cpu") -> State:
    return from_json([json.loads(s)], device)
