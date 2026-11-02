"""Robot state schema, shared by the sampler, teacher, model and ONNX export.

Mirrors `robot.State` in goSysOne (robot/world.go). The feature encoding here
is the contract with any runtime that loads the exported model: it is written
next to the .onnx file as features.json so a Go caller can rebuild the vector.
"""

from dataclasses import dataclass

BEHAVIORS = [
    "continue_task",
    "slow_down",
    "stop_and_wait",
    "return_to_dock",
    "request_human_help",
]
HAZARD_LEVELS = ["none", "low", "moderate", "high", "critical"]
NEEDS_HUMAN = ["false", "true"]

HEADS = {
    "behavior": BEHAVIORS,
    "hazard": HAZARD_LEVELS,
    "needs_human": NEEDS_HUMAN,
}

# goSysOne reports nearest_obstacle_m = 99 when nothing is in range.
NO_OBSTACLE_M = 99.0


@dataclass(frozen=True)
class Feature:
    name: str  # JSON field in robot.State (or derived, see `kind`)
    kind: str  # "num": (clip(x, lo, hi) - lo) / (hi - lo); "bool": 0/1; "onehot": last_behavior == value
    lo: float = 0.0
    hi: float = 1.0
    value: str = ""


def _features() -> list[Feature]:
    f = [
        Feature("battery_pct", "num", 0, 100),
        Feature("speed_mps", "num", -1.5, 1.5),
        Feature("distance_to_goal_m", "num", 0, 60),
        Feature("distance_to_dock_m", "num", 0, 60),
        # Clipped at 20 m: anything farther is "clear" for decision purposes,
        # which also folds the 99 sentinel into the same value.
        Feature("nearest_obstacle_m", "num", 0, 20),
        Feature("human_nearby", "bool"),
        Feature("human_distance_m", "num", 0, 15),
        Feature("human_approaching", "bool"),
        Feature("payload_kg", "num", 0, 40),
        Feature("payload_secure", "bool"),
        Feature("failed_grasps", "num", 0, 5),
        Feature("motor_temp_c", "num", 20, 100),
        Feature("seconds_on_task", "num", 0, 600),
    ]
    f += [Feature("last_behavior", "onehot", value=b) for b in BEHAVIORS]
    return f


FEATURES = _features()
N_FEATURES = len(FEATURES)


def feature_spec() -> dict:
    """JSON-serializable description of the input encoding."""
    return {
        "input": "features",
        "dtype": "float32",
        "features": [
            {"name": f.name, "kind": f.kind, "lo": f.lo, "hi": f.hi, "value": f.value}
            for f in FEATURES
        ],
        "outputs": {f"{k}_probs": v for k, v in HEADS.items()},
    }
