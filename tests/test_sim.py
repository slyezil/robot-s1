import torch

from robot_s1.rollout import lookahead_q, run_episodes, teacher_policy
from robot_s1.sim import CONTINUE, DOCK, HELP, INF, Env, SimConfig

QUIET = SimConfig(person_rate=0, pallet_rate=0, slip_base=0, slip_speed=0)


def env1(cfg=QUIET, **kw) -> Env:
    e = Env.random(1, cfg, torch.Generator().manual_seed(0), "cpu")
    base = dict(pos=0.0, goal=30.0, battery=80.0, temp=40.0, kg=10.0, secure=1.0)
    for k, v in {**base, **kw}.items():
        e.f[k] = torch.tensor([v])
    return e


def run(e: Env, action: int, seconds: float):
    g = torch.Generator().manual_seed(1)
    for _ in range(round(seconds / (e.cfg.dt * e.cfg.decide_every))):
        e.decide(torch.tensor([action]), g)


def test_continue_delivers():
    e = env1()
    run(e, CONTINUE, 30)
    assert e.delivered.item() and e.done.item()
    assert e.ret.item() > 8  # delivery bonus minus ~21 s of time cost


def test_reflex_stops_short_of_pallet():
    e = env1(goal=60.0)
    e.f["obst"], e.f["obst_clear"] = torch.tensor([10.0]), torch.tensor([INF])
    run(e, CONTINUE, 30)
    assert 9.4 < e.pos.item() < 10.0
    assert abs(e.speed.item()) < 0.05


def test_dock_charges():
    e = env1(pos=1.0, battery=5.0)
    run(e, DOCK, 5)
    assert e.pos.item() == 0 and e.battery.item() > 10


def test_unsecured_payload_operator_fixes_it():
    e = env1(secure=0.0)
    run(e, HELP, 4)
    assert e.secure.item() == 1 and e.pages.item() == 1 and not e.dropped.item()


def test_dead_battery_strands():
    e = env1(battery=0.05)
    run(e, CONTINUE, 5)
    assert e.stranded.item() and e.done.item()


def test_lookahead_prefers_help_for_slipped_payload():
    e = env1(secure=0.0)
    q = lookahead_q(e, teacher_policy, torch.Generator().manual_seed(2), k=4, horizon_s=30)
    assert q.argmax().item() == HELP


def test_common_random_numbers():
    cfg = SimConfig()
    a = run_episodes(Env.random(64, cfg, torch.Generator().manual_seed(5), "cpu"), teacher_policy,
                     torch.Generator().manual_seed(6))
    b = run_episodes(Env.random(64, cfg, torch.Generator().manual_seed(5), "cpu"), teacher_policy,
                     torch.Generator().manual_seed(6))
    assert torch.equal(a.ret, b.ret)
