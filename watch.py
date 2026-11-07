"""Watch one robot through one episode, decision by decision.

    python watch.py                              # improved model
    python watch.py --policy heuristic --seed 7
    python watch.py --policy plan                # lookahead planner picks every move (slow)

Prints a line whenever the behavior changes or something happens in the world.
`--policy plan` controls the robot with the simulator itself: at each decision
it copies the world, plays each behavior forward, and takes the best one. That
is the slow "System 2" the fast model is trained to imitate.
"""

import argparse
import math
from pathlib import Path

import torch

from export import load
from robot_s1.rollout import act, lookahead_q, net_policy, rules_policy, teacher_policy
from robot_s1.schema import BEHAVIORS
from robot_s1.sim import Env, SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="model", choices=["model", "distilled", "heuristic", "rules", "plan"])
    ap.add_argument("--ckpt", type=Path, default=Path("runs/improved/model.pt"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--plan-k", type=int, default=4)
    ap.add_argument("--plan-horizon", type=float, default=20.0)
    args = ap.parse_args()

    dev = torch.device("cpu")  # one robot: the CPU is faster than the GPU here
    cfg = SimConfig()
    model = None
    if args.policy in ("model", "plan"):
        model = net_policy(load(args.ckpt))
    elif args.policy == "distilled":
        model = net_policy(load(Path("runs/latest/model.pt")))
    policy = {"heuristic": teacher_policy, "rules": rules_policy}.get(args.policy, model)

    env = Env.random(1, cfg, torch.Generator().manual_seed(args.seed), dev)
    gen = torch.Generator().manual_seed(args.seed + 1)
    e = env.f
    print(f"goal {e['goal'].item():.1f} m, start at {e['pos'].item():.1f} m, battery {e['battery'].item():.0f}%, "
          f"payload {e['kg'].item():.0f} kg {'secure' if e['secure'].item() else 'UNSECURED'}, motor {e['temp'].item():.0f} C")
    print(f"{'t':>6} {'pos':>6} {'speed':>5} {'batt':>5} {'motor':>5}  {'payload':8} {'person':>6} {'obst':>5}  behavior")

    prev, prev_events = None, None
    while not bool(env.done.all()):
        s = env.sense()
        if args.policy == "plan":
            q = lookahead_q(env, model, gen, k=args.plan_k, horizon_s=args.plan_horizon)
            a = q.argmax(-1)
            extra = "  scores " + " ".join(f"{b.split('_')[0]}={v:.1f}" for b, v in zip(BEHAVIORS, q[0].tolist()))
        else:
            a = act(policy(s), gen)
            extra = ""
        events = (math.isfinite(e["human"].item()), math.isfinite(e["obst"].item()), e["secure"].item() > 0.5)
        if a.item() != prev or events != prev_events:
            person = f"{s['human_distance_m'].item():.1f}" if s["human_nearby"].item() else "-"
            obst = s["nearest_obstacle_m"].item()
            print(f"{e['t'].item():6.2f} {e['pos'].item():6.1f} {e['speed'].item():5.2f} {e['battery'].item():5.1f} "
                  f"{e['temp'].item():5.1f}  {'secure' if events[2] else 'SLIPPED':8} {person:>6} "
                  f"{('-' if obst >= 99 else f'{obst:.1f}'):>5}  {BEHAVIORS[a.item()]}{extra}")
            prev, prev_events = a.item(), events
        env.decide(a, gen, max_time=cfg.max_time)

    outcome = [k for k in ("delivered", "stranded", "dropped", "fault", "timeout") if e[k].item()]
    print(f"\n{outcome[0] if outcome else '?'} at t={e['t'].item():.1f}s  return {e['ret'].item():.2f}  "
          f"exposure {e['exposure'].item():.2f}  pages {int(e['pages'].item())}")


if __name__ == "__main__":
    main()
