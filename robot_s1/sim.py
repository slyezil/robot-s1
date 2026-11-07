"""Time-stepped port of goSysOne's corridor world (robot/world.go) and fast
controller (robot/control.go), so we can play out what happens *after* a
decision: the next state, and the next, until the robot delivers or fails.

Each row of an Env is one independent robot timeline. Rows exist so the same
code can run one robot (watch.py) or many what-if copies of one moment
(lookahead in improve.py). Done rows are frozen.

Differences from goSysOne, all deliberate:
- The controller runs at 1/dt (20 Hz) instead of 1 kHz.
- Disturbances are random events with per-second rates instead of a fixed
  script, and the payload can slip or fall depending on speed and weight.
- Outcomes are scored (see SimConfig rewards) so behaviors can be compared.
"""

from dataclasses import dataclass

import torch

from .schema import BEHAVIORS, NO_OBSTACLE_M
from .states import State

CONTINUE, SLOW, STOP, DOCK, HELP = range(len(BEHAVIORS))
INF = float("inf")


@dataclass
class SimConfig:
    dt: float = 0.05
    decide_every: int = 5  # physics steps per supervisor decision (0.25 s, as in goSysOne)
    max_speed: float = 1.5
    max_time: float = 150.0  # episode timeout, seconds

    # goSysOne world constants
    charge_per_s: float = 8.0
    operator_delay: float = 3.0
    human_speed: float = 0.6
    battery_drain_scale: float = 1.0  # multiplies goSysOne's drain (0.02 + 0.05*|v|) %/s

    # Random disturbances (rates per second)
    person_rate: float = 1 / 20
    pallet_rate: float = 1 / 30
    pallet_clear_s: tuple[float, float] = (4.0, 10.0)
    slip_base: float = 0.001  # payload slips on its own...
    slip_speed: float = 0.004  # ...plus this * speed^2 * kg/20
    drop_rate: float = 0.15  # an unsecured payload falls off at this * |speed|
    motor_fault_c: float = 85.0

    # Rewards. These encode what "good" means; change them and the learned policy changes.
    r_deliver: float = 10.0
    r_time: float = -0.05  # per second
    r_stranded: float = -20.0  # battery hit 0 away from the dock
    r_dropped: float = -15.0
    r_fault: float = -15.0  # motor overheated
    r_page: float = -1.0  # each operator call
    r_exposure: float = -1.0  # * speed^2 per second with a person within exposure_m
    exposure_m: float = 4.0


FIELDS = [
    "pos", "goal", "speed", "battery", "temp", "kg", "secure",
    "obst", "obst_clear", "human", "operator", "t",
    "behavior", "last", "done",
    "ret", "delivered", "stranded", "dropped", "fault", "timeout", "exposure", "pages",
    "noise",  # rows with the same noise id see the same random events (common random numbers)
]


class Env:
    def __init__(self, cfg: SimConfig, **f: torch.Tensor):
        self.cfg = cfg
        self.f = f

    def __getattr__(self, k):
        f = self.__dict__.get("f")
        if f is not None and k in f:
            return f[k]
        raise AttributeError(k)

    @property
    def n(self) -> int:
        return self.f["pos"].shape[0]

    @property
    def device(self):
        return self.f["pos"].device

    # ---- construction ----

    @classmethod
    def random(cls, n: int, cfg: SimConfig, gen: torch.Generator, device) -> "Env":
        def u(lo, hi):
            return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)

        goal = u(15, 60)
        low = torch.rand(n, generator=gen, device=device) < 0.4
        z = torch.zeros(n, device=device)
        inf = torch.full((n,), INF, device=device)
        return cls(
            cfg,
            pos=goal * u(0, 0.6),
            goal=goal,
            speed=z.clone(),
            battery=torch.where(low, u(1, 30), u(30, 100)),
            temp=u(30, 70),
            kg=u(0, 40),
            secure=(torch.rand(n, generator=gen, device=device) > 0.1).float(),
            obst=inf.clone(),
            obst_clear=inf.clone(),
            human=inf.clone(),
            operator=inf.clone(),
            t=z.clone(),
            behavior=torch.full((n,), CONTINUE, device=device, dtype=torch.long),
            last=torch.full((n,), CONTINUE, device=device, dtype=torch.long),
            done=torch.zeros(n, dtype=torch.bool, device=device),
            ret=z.clone(),
            delivered=torch.zeros(n, dtype=torch.bool, device=device),
            stranded=torch.zeros(n, dtype=torch.bool, device=device),
            dropped=torch.zeros(n, dtype=torch.bool, device=device),
            fault=torch.zeros(n, dtype=torch.bool, device=device),
            timeout=torch.zeros(n, dtype=torch.bool, device=device),
            exposure=z.clone(),
            pages=z.clone(),
            noise=torch.arange(n, device=device),
        )

    def select(self, idx: torch.Tensor) -> "Env":
        return Env(self.cfg, **{k: v[idx] for k, v in self.f.items()})

    def repeat_interleave(self, k: int) -> "Env":
        return Env(self.cfg, **{k_: v.repeat_interleave(k, 0) for k_, v in self.f.items()})

    @staticmethod
    def cat(envs: list["Env"]) -> "Env":
        return Env(envs[0].cfg, **{k: torch.cat([e.f[k] for e in envs]) for k in FIELDS})

    # ---- sensing (mirrors World.Sense) ----

    def _ranges(self):
        d_obst = self.obst - self.pos
        obst_m = torch.where(d_obst > -0.5, d_obst.clamp(min=0), torch.full_like(d_obst, NO_OBSTACLE_M))
        obst_m = obst_m.clamp(max=NO_OBSTACLE_M)
        d_h = self.human - self.pos
        near = (d_h > -1) & (d_h < 15)
        h_m = torch.where(near, d_h.clamp(min=0), torch.zeros_like(d_h))
        nearest = torch.where(near, torch.minimum(obst_m, h_m), obst_m)
        return nearest, near, h_m

    def sense(self) -> State:
        nearest, near, h_m = self._ranges()
        nf = near.float()
        return {
            "battery_pct": self.battery,
            "speed_mps": self.speed,
            "distance_to_goal_m": self.goal - self.pos,
            "distance_to_dock_m": self.pos,
            "nearest_obstacle_m": nearest,
            "human_nearby": nf,
            "human_distance_m": h_m,
            "human_approaching": nf,
            "payload_kg": self.kg,
            "payload_secure": self.secure,
            "failed_grasps": torch.zeros_like(self.pos),
            "motor_temp_c": self.temp,
            "seconds_on_task": self.t,
            "last_behavior": self.last.float(),
        }

    # ---- dynamics ----

    def decide(self, action: torch.Tensor, gen: torch.Generator, max_time: float | None = None):
        """Apply one supervisor decision, then run physics until the next one."""
        c = self.cfg
        alive = ~self.done
        page = alive & (action == HELP) & (self.secure < 0.5) & torch.isinf(self.operator)
        self.f["operator"] = torch.where(page, self.t + c.operator_delay, self.operator)
        self.f["pages"] = self.pages + page.float()
        self.f["ret"] = self.ret + page.float() * c.r_page
        self.f["behavior"] = torch.where(alive, action, self.behavior)
        self.f["last"] = self.behavior.clone()
        for _ in range(c.decide_every):
            self._physics(gen)
        if max_time is not None:
            to = ~self.done & (self.t >= max_time)
            self.f["timeout"] = self.timeout | to
            self.f["done"] = self.done | to

    def _physics(self, gen: torch.Generator):
        c, dt, n, dev = self.cfg, self.cfg.dt, self.n, self.device
        old = dict(self.f)
        f = dict(self.f)
        # Drawn every step whether or not a row is alive, so the same seed gives
        # every policy the same event stream; rows sharing a noise id share draws.
        ids = f["noise"]
        r = torch.rand(8, int(ids.max()) + 1, generator=gen, device=dev)[:, ids]

        # Controller (control.go): target speed, reflex envelope, 1 m/s^2 accel limit.
        b = f["behavior"]
        target = torch.zeros(n, device=dev)
        target = torch.where(b == CONTINUE, c.max_speed, target)
        target = torch.where(b == SLOW, 0.3 * c.max_speed, target)
        target = torch.where(b == DOCK, -0.6 * c.max_speed, target)
        nearest, _, _ = self._ranges()
        fwd = target > 0
        target = torch.where(fwd & (nearest < 0.5), 0.0, target)
        target = torch.where(fwd & (nearest >= 0.5) & (nearest < 2), torch.minimum(target, 0.2 * nearest), target)
        speed = f["speed"] + (target - f["speed"]).clamp(-dt, dt)

        # World.Step
        pos = (f["pos"] + speed * dt).clamp(min=0)
        pos = torch.minimum(pos, f["goal"])
        charging = (pos == 0) & (speed <= 0)
        drain = dt * (0.02 + 0.05 * speed.abs()) * c.battery_drain_scale
        battery = torch.where(charging, (f["battery"] + dt * c.charge_per_s).clamp(max=100), (f["battery"] - drain).clamp(min=0))
        temp = f["temp"] + dt * (0.4 * speed.abs() * f["kg"] / 10 - 0.05 * (f["temp"] - 35))
        t = f["t"] + dt

        human = f["human"] - dt * c.human_speed
        human = torch.where((human - pos < -1) | (human < 0), INF, human)
        cleared = t >= f["obst_clear"]
        obst = torch.where(cleared, INF, f["obst"])
        obst_clear = torch.where(cleared, INF, f["obst_clear"])
        arrived = t >= f["operator"]
        secure = torch.where(arrived, 1.0, f["secure"])
        operator = torch.where(arrived, INF, f["operator"])

        # Random disturbances
        spawn = torch.isinf(human) & (r[0] < c.person_rate * dt)
        human = torch.where(spawn, torch.minimum(pos + 8 + 7 * r[1], f["goal"]), human)
        drop_pallet = torch.isinf(obst) & (r[2] < c.pallet_rate * dt)
        obst = torch.where(drop_pallet, torch.minimum(pos + 3 + 7 * r[3], f["goal"] - 0.5), obst)
        lo, hi = c.pallet_clear_s
        obst_clear = torch.where(drop_pallet, t + lo + (hi - lo) * r[4], obst_clear)
        slip_p = (c.slip_base + c.slip_speed * speed**2 * f["kg"] / 20) * dt
        secure = torch.where((secure > 0.5) & (r[5] < slip_p), 0.0, secure)
        dropped = (secure < 0.5) & (r[6] < c.drop_rate * speed.abs() * dt)

        # Scoring
        d_h = human - pos
        exposed = (d_h > -1) & (d_h < c.exposure_m)
        exp_cost = exposed.float() * speed**2 * dt
        delivered = pos >= f["goal"]
        stranded = (battery <= 0) & ~charging
        fault = temp >= c.motor_fault_c
        ret = (
            f["ret"] + c.r_time * dt + c.r_exposure * exp_cost
            + delivered.float() * c.r_deliver + stranded.float() * c.r_stranded
            + dropped.float() * c.r_dropped + fault.float() * c.r_fault
        )

        new = dict(
            pos=pos, speed=speed, battery=battery, temp=temp, t=t, human=human, obst=obst,
            obst_clear=obst_clear, secure=secure, operator=operator, ret=ret,
            exposure=f["exposure"] + exp_cost,
            delivered=f["delivered"] | delivered, stranded=f["stranded"] | stranded,
            dropped=f["dropped"] | dropped, fault=f["fault"] | fault,
        )
        new["done"] = f["done"] | delivered | stranded | dropped | fault
        alive = ~old["done"]
        for k, v in new.items():
            self.f[k] = torch.where(alive, v, old[k])
