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

## Roadmap

1. Distill the goSysOne heuristic (this step).
2. Load the ONNX model as a `decision.Decider` backend in goSysOne.
3. Go beyond the teacher: label edge cases by hand, record simulation runs,
   and score episodes by outcome (deliveries, near misses, battery deaths)
   instead of imitating the heuristic.
