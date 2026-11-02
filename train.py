"""Train the small decision model on the GPU by distilling the teacher.

Every step samples a fresh batch of random states on the GPU and labels it
with the vectorized teacher, so there is no dataset on disk and no overfitting
to a fixed set.

    python train.py                     # defaults, ~7 min on an MX350 (peak ~450 MiB)
    python train.py --steps 40000 --hidden 512
"""

import argparse
import math
import time
from pathlib import Path

import torch

from robot_s1.model import DecisionNet
from robot_s1.schema import HEADS
from robot_s1.states import encode, sample_states
from robot_s1.teacher import teacher_probs


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * logits.log_softmax(-1)).sum(-1).mean()


@torch.no_grad()
def evaluate(net, x, targets) -> dict[str, float]:
    net.eval()
    out = net(x)
    net.train()
    m = {}
    for k in HEADS:
        p, t = out[k].softmax(-1), targets[k]
        kl = (t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean()
        m[f"{k}/kl"] = kl.item()
        m[f"{k}/top1"] = (p.argmax(-1) == t.argmax(-1)).float().mean().item()
        m[f"{k}/max_err"] = (p - t).abs().max().item()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    ap.add_argument("--cpu", action="store_true", help="train on CPU even if a GPU is available")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    torch.manual_seed(args.seed)

    # Fixed held-out set from a different seed.
    eval_gen = torch.Generator(device=dev).manual_seed(10_000 + args.seed)
    eval_states = sample_states(200_000, eval_gen, dev)
    eval_x, eval_t = encode(eval_states), teacher_probs(eval_states)

    net = DecisionNet(args.hidden, args.layers).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(500, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        s = sample_states(args.batch, gen, dev)
        x, t = encode(s), teacher_probs(s)
        out = net(x)
        loss = sum(soft_ce(out[k], t[k]) for k in HEADS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(net, eval_x, eval_t)
            el = time.time() - t0
            print(
                f"step {step:6d}  loss {loss.item():.4f}  "
                + "  ".join(f"{k}: kl {m[k + '/kl']:.5f} top1 {m[k + '/top1']:.4f}" for k in HEADS)
                + f"  [{el:.0f}s, {step * args.batch / el / 1e6:.1f}M states/s]"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "model.pt"
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "layers": args.layers,
                "metrics": evaluate(net, eval_x, eval_t)}, ckpt)
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
