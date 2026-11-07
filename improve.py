"""Improve the decision model by simulating what happens after each decision.

Each iteration:
  1. Run episodes with the current model (plus some exploration) and save
     snapshots of moments the robot actually reached.
  2. For every snapshot, try each of the 5 behaviors: copy the world, commit to
     that behavior for 1 s, then let the current model drive for the next 60 s
     of simulated states. Score the outcome (delivered, stranded, dropped
     payload, motor fault, speed near people, time). Average over k copies
     with different random events.
  3. Train the behavior head toward behaviors that did better than the
     alternatives: target = current probs * exp((Q - max Q) / tau). Ties keep
     the current choice, so noise cannot push the robot into arbitrary
     behaviors. The hazard and needs_human heads keep learning
     from the heuristic teacher.
  4. Evaluate on a fixed set of episodes against the if/else rules and the
     heuristic.

    python improve.py                         # from runs/latest -> runs/improved
    python improve.py --iters 8 --snapshots 4096
"""

import argparse
import json
import time
from pathlib import Path

import torch

from export import load
from robot_s1.rollout import (lookahead_q, net_policy, rules_policy, run_episodes, summarize,
                              teacher_policy)
from robot_s1.schema import HEADS
from robot_s1.sim import Env, SimConfig
from robot_s1.states import encode, sample_states
from robot_s1.teacher import teacher_probs
from train import soft_ce


def collect(net, cfg, n_episodes, max_snaps, explore, gen, dev) -> Env:
    """Run episodes with the current model; keep random moments as snapshots."""
    snaps, count = [], 0

    def hook(env: Env, _a):
        nonlocal count
        pick = ~env.done & (torch.rand(env.n, generator=gen, device=dev) < 0.03)
        idx = pick.nonzero().squeeze(-1)
        if idx.numel():
            snaps.append(env.select(idx))
            count += idx.numel()

    env = Env.random(n_episodes, cfg, gen, dev)
    net.eval()
    run_episodes(env, net_policy(net), gen, explore=explore, on_decision=hook)
    s = Env.cat(snaps)
    if s.n > max_snaps:
        s = s.select(torch.randperm(s.n, generator=gen, device=dev)[:max_snaps])
    # Reset the bookkeeping so lookahead scores only the future.
    for k in ("delivered", "stranded", "dropped", "fault", "timeout"):
        s.f[k] = torch.zeros_like(s.f[k])
    return s


def evaluate(policy, cfg, n, seed, dev) -> dict:
    env = Env.random(n, cfg, torch.Generator(device=dev).manual_seed(seed), dev)
    run_episodes(env, policy, torch.Generator(device=dev).manual_seed(seed + 1))
    return summarize(env)


def fmt(m: dict) -> str:
    return (f"return {m['return']:6.2f}  delivered {m['delivered%']:5.1f}%  stranded {m['stranded%']:4.1f}%  "
            f"dropped {m['dropped%']:4.1f}%  fault {m['fault%']:4.1f}%  timeout {m['timeout%']:4.1f}%  "
            f"time {m['time_s']:5.1f}s  exposure {m['exposure']:.2f}  pages {m['pages']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, default=Path("runs/latest/model.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/improved"))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=1024, help="episodes per collection round")
    ap.add_argument("--snapshots", type=int, default=3072, help="moments to look ahead from per round")
    ap.add_argument("--k", type=int, default=8, help="random futures per (moment, behavior)")
    ap.add_argument("--horizon", type=float, default=60.0, help="lookahead seconds")
    ap.add_argument("--tau", type=float, default=1.0, help="how strongly score differences move the policy")
    ap.add_argument("--explore", type=float, default=0.2)
    ap.add_argument("--train-steps", type=int, default=1500)
    ap.add_argument("--eval-episodes", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    cfg = SimConfig()
    net = load(args.init).to(dev)
    ckpt = torch.load(args.init, map_location="cpu")
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    eval_seed = 1_000_000

    print("baselines on the fixed evaluation episodes:")
    for name, pol in [("rules", rules_policy), ("heuristic", teacher_policy)]:
        print(f"  {name:10s} {fmt(evaluate(pol, cfg, args.eval_episodes, eval_seed, dev))}")
    net.eval()
    best = evaluate(net_policy(net), cfg, args.eval_episodes, eval_seed, dev)
    print(f"  {'iter 0':10s} {fmt(best)}")
    history = [{"iter": 0, **best}]
    args.out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    for it in range(1, args.iters + 1):
        t0 = time.time()
        snaps = collect(net, cfg, args.episodes, args.snapshots, args.explore, gen, dev)
        t1 = time.time()
        q = lookahead_q(snaps, net_policy(net), gen, k=args.k, horizon_s=args.horizon)
        t2 = time.time()
        xs.append(encode(snaps.sense()))
        with torch.no_grad():
            prior = net(xs[-1])["behavior"].softmax(-1)
        target = prior * ((q - q.max(-1, keepdim=True).values) / args.tau).exp()
        ys.append(target / target.sum(-1, keepdim=True))
        x, y = torch.cat(xs), torch.cat(ys)
        changed = (prior.argmax(-1) != ys[-1].argmax(-1)).float().mean().item()

        net.train()
        for _ in range(args.train_steps):
            i = torch.randint(0, x.shape[0], (2048,), generator=gen, device=dev)
            aux = sample_states(2048, gen, dev)
            t_aux = teacher_probs(aux)
            out, out_aux = net(x[i]), net(encode(aux))
            loss = soft_ce(out["behavior"], y[i]) + sum(
                soft_ce(out_aux[k], t_aux[k]) for k in HEADS if k != "behavior")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        t3 = time.time()

        m = evaluate(net_policy(net), cfg, args.eval_episodes, eval_seed, dev)
        history.append({"iter": it, **m})
        print(f"  {'iter ' + str(it):10s} {fmt(m)}")
        print(f"             {snaps.n} moments, target's top behavior changed in {100 * changed:.0f}%"
              f"  [collect {t1 - t0:.0f}s, lookahead {t2 - t1:.0f}s, train {t3 - t2:.0f}s, eval {time.time() - t3:.0f}s]")
        if m["return"] > best["return"]:
            best = m
            torch.save({**ckpt, "state_dict": net.state_dict(), "metrics": m}, args.out / "model.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best: {fmt(best)}")
    print(f"saved {args.out / 'model.pt'}" if (args.out / "model.pt").exists() else "no iteration beat the start")


if __name__ == "__main__":
    main()
