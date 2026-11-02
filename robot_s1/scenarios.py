"""Named states from the goSysOne demo script, used as a sanity check that the
student agrees with the teacher where it matters most."""

BASE = {
    "task": "deliver tote to station B",
    "battery_pct": 38,
    "speed_mps": 1.5,
    "distance_to_goal_m": 30,
    "distance_to_dock_m": 10,
    "nearest_obstacle_m": 99,
    "human_nearby": False,
    "payload_kg": 12,
    "payload_secure": True,
    "failed_grasps": 0,
    "motor_temp_c": 41,
    "seconds_on_task": 7,
    "last_behavior": "continue_task",
}


def _s(**kw):
    return {**BASE, **kw}


SCENARIOS = {
    "cruising": _s(),
    "person 10m approaching": _s(human_nearby=True, human_distance_m=10, human_approaching=True, nearest_obstacle_m=10),
    "person 2m approaching": _s(human_nearby=True, human_distance_m=2, human_approaching=True, nearest_obstacle_m=2),
    "pallet 1m ahead": _s(nearest_obstacle_m=1, speed_mps=0.3),
    "payload slipped": _s(payload_secure=False),
    "low battery, 3m from goal": _s(battery_pct=12, distance_to_goal_m=3, distance_to_dock_m=37),
    "low battery, 35m from goal": _s(battery_pct=12, distance_to_goal_m=35, distance_to_dock_m=5),
    "hot motor": _s(motor_temp_c=78),
    "repeated failed grasps": _s(failed_grasps=3, speed_mps=0),
}
