# robot-s1

A small, fast "System One" decision model for the goSysOne warehouse robot,
trained on a 2 GB laptop GPU (GeForce MX350).

goSysOne asks three questions about the robot state every supervisor tick:
`behavior` (5 options), `hazard` (5 levels) and `needs_human` (bool). Its
current backends are a hand-weighted heuristic and Laya, a 421M-parameter
model that needs fine-tuning and does not fit this GPU comfortably. This
project trains a fixed-schema multi-head MLP instead: one forward pass answers
all three questions in microseconds.

## How it trains

- `robot_s1/states.py`: samples random robot states directly on the GPU,
  with more combinations than the scripted demo produces (low battery plus a
  person close by plus a hot motor, and so on).
- `robot_s1/teacher.py`: a vectorized port of goSysOne's `robot/heuristic.go`
  that labels each state with full probability distributions.
- `train.py`: distills the teacher into the student using a fresh batch of
  states every step, so no dataset is stored on disk.
- `export.py`: writes `robot_s1.onnx` and `features.json`, which describes the
  input encoding a Go caller needs to rebuild, and checks onnxruntime's output
  against PyTorch.
- `evaluate.py`: compares student and teacher on the goSysOne demo scenarios
  and measures latency.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install onnx onnxruntime numpy pytest
```

Use the **cu126** wheels. Newer CUDA builds of PyTorch dropped Pascal (sm_61)
GPUs such as the MX350.

## Run

```powershell
.venv\Scripts\python -m pytest -q     # teacher matches the Go heuristic
.venv\Scripts\python train.py         # trains on the GPU -> runs/latest/model.pt
.venv\Scripts\python evaluate.py      # scenario agreement + latency
.venv\Scripts\python export.py        # -> runs/latest/robot_s1.onnx
```

## Improving by simulating ahead

`robot_s1/sim.py` ports goSysOne's corridor world and 1 kHz controller,
including its reflexes, to a time-stepped simulator. It adds random people,
pallets, payload slips, motor heat and battery, and scores each episode:
delivered (+10), time (-0.05/s), stranded (-20), dropped payload (-15),
motor fault (-15), operator page (-1), and speed near people (-speed^2 per
second). These weights in `SimConfig` define what "good" means.

`improve.py` runs one round at a time:
1. The current model drives episodes, and random moments are saved.
2. From each moment, every behavior is tried for 1 s, then the model drives
   for the next 60 s. All behaviors see the same 8 random futures, so score
   differences come from the choice of behavior rather than luck.
3. The model is nudged toward behaviors that scored better
   (`target = current probs * exp(advantage / tau)`). Where the scores tie, it
   keeps its current choice.

```powershell
.venv\Scripts\python improve.py              # ~12 min, writes runs/improved/model.pt
.venv\Scripts\python watch.py                # one robot, decision by decision
.venv\Scripts\python watch.py --policy plan  # the lookahead planner drives instead (slow)
```

## Running on the edge

`robot_s1/edge.py` is the inference path for one robot making one decision at
a time. It preallocates pinned host and device buffers and replays the whole
forward pass (all three heads plus softmax) as a single CUDA graph, so a
decision costs one 72-byte copy in, one graph launch, and one 48-byte copy
out. `bench_edge.py` measures it; `sweep_edge.py` shows where the GPU
overtakes a CPU core as the model grows.

```powershell
.venv\Scripts\python bench_edge.py     # p50/p99 latency per decision, deadline misses, closed loop
.venv\Scripts\python sweep_edge.py     # CPU onnxruntime vs CUDA graph across model sizes
```

Measured on the MX350 (laptop, Windows, no locked clocks):

| model | params | CPU onnxruntime p50 | CUDA graph p50 |
|---|---|---|---|
| 256 x 3 (this one) | 140k | 66-74 us | 190-325 us |
| 512 x 4 | 0.8M | 290 us | 455 us |
| 1024 x 4 | 3.2M | 1200 us | 580 us |
| 2048 x 4 | 12.7M | 3500 us | 2000 us |

Below about 1M parameters a decision is cheaper on one CPU core: the GPU's
floor is the PCIe round trip, ~200 us. Above that the GPU wins and keeps a
tighter p99. Naive per-call PyTorch on the GPU is ~1 ms and misses a 1 kHz
deadline 94% of the time; the CUDA graph is what makes it viable. fp16 does
not help on Pascal.

## Roadmap

1. Distill the goSysOne heuristic (this step).
2. Load the ONNX model as a `decision.Decider` backend in goSysOne.
3. Improve on the teacher by simulating ahead (done, see above).
