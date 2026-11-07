"""Policies, episode runner, and lookahead planner.

A policy maps a batch of sensed states to behavior probabilities [N, 5].
"""

from collections.abc import Callable

import torch

from .model import DecisionNet
from .schema import BEHAVIORS
from .sim import CONTINUE, DOCK, HELP, SLOW, STOP, Env
from .states import State, encode
from .teacher import teacher_probs

Policy = Callable[[State], torch.Tensor]


def net_policy(net: DecisionNet) -> Policy:
    @torch.no_grad()
    def p(s: State) -> torch.Tensor:
        return net(encode(s))["behavior"].softmax(-1)

    return p


def teacher_policy(s: State) -> torch.Tensor:
    return teacher_probs(s)["behavior"]


def rules_policy(s: State) -> torch.Tensor:
    """goSysOne robot/rules.go: first matching branch wins."""
    a = torch.full_like(s["battery_pct"], CONTINUE, dtype=torch.long)
    a = torch.where(s["human_nearby"] > 0.5, SLOW, a)
    a = torch.where(s["nearest_obstacle_m"] < 3, STOP, a)
    a = torch.where(s["battery_pct"] < 20, DOCK, a)
    a = torch.where(s["payload_secure"] < 0.5, HELP, a)
    return torch.nn.functional.one_hot(a, len(BEHAVIORS)).float()


def act(probs: torch.Tensor, gen: torch.Generator, explore: float = 0.0) -> torch.Tensor:
    """Greedy when explore == 0; otherwise sample, mixing in `explore` uniform."""
    if explore == 0:
        return probs.argmax(-1)
    p = (1 - explore) * probs + explore / probs.shape[-1]
    return torch.multinomial(p, 1, generator=gen).squeeze(-1)


def run_episodes(env: Env, policy: Policy, gen: torch.Generator, explore: float = 0.0,
                 on_decision: Callable[[Env, torch.Tensor], None] | None = None) -> Env:
    """Run every row to completion (or the episode timeout)."""
    while not bool(env.done.all()):
        a = act(policy(env.sense()), gen, explore)
        if on_decision is not None:
            on_decision(env, a)
        env.decide(a, gen, max_time=env.cfg.max_time)
    return env


def summarize(env: Env) -> dict[str, float]:
    d = env.delivered.float()
    return {
        "return": env.ret.mean().item(),
        "delivered%": 100 * d.mean().item(),
        "stranded%": 100 * env.stranded.float().mean().item(),
        "dropped%": 100 * env.dropped.float().mean().item(),
        "fault%": 100 * env.fault.float().mean().item(),
        "timeout%": 100 * env.timeout.float().mean().item(),
        "time_s": (env.t * d).sum().item() / max(1.0, d.sum().item()),
        "exposure": env.exposure.mean().item(),
        "pages": env.pages.mean().item(),
    }


@torch.no_grad()
def lookahead_q(snap: Env, policy: Policy, gen: torch.Generator, k: int = 8,
                horizon_s: float = 60.0, commit_s: float = 1.0) -> torch.Tensor:
    """For each row of `snap`, estimate the value of each behavior [S, 5].

    Copies each moment 5*k times, commits to one behavior for `commit_s`, then
    follows `policy` greedily for the rest of the horizon. Random events differ
    between the k copies, so Q averages over what might happen next. Unfinished
    copies are credited an optimistic estimate of finishing the delivery, the
    same for every behavior.
    """
    a_n = len(BEHAVIORS)
    s_n = snap.n
    env = snap.repeat_interleave(a_n * k)
    first = torch.arange(a_n, device=env.device).repeat_interleave(k).repeat(s_n)
    # Copy j of every behavior sees the same random future, so differences in
    # Q come from the behavior, not from luck.
    dev = env.device
    env.f["noise"] = (torch.arange(s_n, device=dev).repeat_interleave(a_n * k) * k
                      + torch.arange(k, device=dev).repeat(s_n * a_n))
    ret0 = env.ret.clone()
    c = env.cfg
    dec_s = c.dt * c.decide_every
    commit = round(commit_s / dec_s)
    for i in range(round(horizon_s / dec_s)):
        if bool(env.done.all()):
            break
        a = first if i < commit else act(policy(env.sense()), gen)
        env.decide(a, gen)
    tail = (~env.done).float() * (c.r_deliver + c.r_time * (env.goal - env.pos) / c.max_speed)
    return (env.ret - ret0 + tail).view(s_n, a_n, k).mean(-1)
