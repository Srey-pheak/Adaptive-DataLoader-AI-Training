# Adaptive DataLoader for AI Training on HPC

> A throughput‑driven adaptive DataLoader that automatically tunes PyTorch DataLoader configuration during training , eliminating manual tuning and handling HPC deadlocks.

---

## Overview

Standard PyTorch training uses a **static DataLoader configuration**. You set `batch_size`, `num_workers`, and `prefetch_factor` at the start and they never change. This creates three problems:

- **GPU Starvation** — workers too slow → GPU idles between batches → wasted compute
- **GPU Overload** — batch too large → memory pressure → OOM errors
- **Manual Tuning** — finding optimal settings requires hours or days of trial and error

This project solves all three by implementing a **three-layer adaptive feedback loop** that monitors **throughput** (images/second) in real time and automatically adjusts the data pipeline — without stopping training or changing the training algorithm.

### Why Throughput, Not GPU Utilisation?

GPU utilisation is unreliable for lightweight models like AlexNet on an A100. Batches complete in under a millisecond, while `pynvml` samples once per second — reported utilisation never exceeds 30%. **Throughput (images/second)** directly measures end-to-end pipeline efficiency and responds clearly to every configuration change.

---

## System Architecture


## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Process                         │
│                                                             │
│   Training Loop                                             │
│   for batch in AdaptiveDataLoader:                          │
│       model(batch) → loss → backward → step                 │
│              │                                              │
│              │ record_batch(n)                              │
│              ▼                                              │
│   ┌─────────────────────┐                                   │
│   │   GPU Monitor        │  ← Layer 1 — MEASURE             │
│   │   (thread, 1s)       │                                  │
│   │   pynvml + nvidia-smi│                                  │
│   │   → rolling averages │                                  │
│   └──────────┬──────────┘                                   │
│              │ GPUStats                                     │
│              ▼                                              │
│   ┌─────────────────────┐                                   │
│   │ Adaptive Controller  │  ← Layer 2 — DECIDE             │
│   │   (thread, 2min)     │                                  │
│   │   FSM decision logic │                                  │
│   │   cooldown 60s       │                                  │
│   └──────────┬──────────┘                                   │
│              │ writes                                       │
│              ▼                                              │
│   ┌─────────────────────┐                                   │
│   │     config.py        │  ← Shared State                 │
│   │   batch_size         │                                  │
│   │   num_workers        │                                  │
│   │   prefetch_factor    │                                  │
│   └──────────┬──────────┘                                   │
│              │ reads                                        │
│              ▼                                              │
│   ┌─────────────────────┐                                   │
│   │ Adaptive DataLoader  │  ← Layer 3 — ACT                │
│   │   wraps DataLoader   │                                  │
│   │   rebuilds on change │                                  │
│   └─────────────────────┘                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
my_advanced_computing_project/
│
├── adaptive_dataloader/          ← The adaptive library
│   ├── config.py                 ← All settings (single source of truth)
│   ├── monitor.py                ← Layer 1: GPU metrics sampler
│   ├── controller.py             ← Layer 2: FSM decision logic
│   └── loader.py                 ← Layer 3: Adaptive DataLoader wrapper
│
├── scripts/
│   ├── train_baseline.py         ← Fixed config training + profiler
│   └── train_adaptive.py         ← Adaptive training using all 3 layers
│
├── jobs/
│   ├── baseline_job.sh           ← SLURM baseline submission
│   └── adaptive_job.sh           ← SLURM adaptive submission
│
├── data/
│   └── imagenet/
│       ├── train/                ← 1,281,167 images, 1000 classes
│       └── val/                  ← 50,000 images, 1000 classes
│
├── logs/
│   ├── baseline_alexnet_metrics.csv   ← per-epoch metrics
│   ├── adaptive_alexnet_metrics.csv   ← per-epoch metrics
│   └── profile_traces/               ← torch.profiler trace files only baseline
│
├── results/
│   └── plots/                    ← result visualizations
│
└── requirements.txt
```

---




---

## How It Works

### Layer 1 — Monitor (`monitor.py`)

Background thread sampling GPU metrics every second via `pynvml`:

- GPU utilisation (%)
- VRAM used / free (MB)
- Throughput (images/second)

Maintains a **30‑second rolling average** for stable readings.

### Layer 2 — Controller (`controller.py`)

Wakes up every **60 seconds** and implements **throughput‑driven coordinate descent**:

| Phase | Action | Stop Condition |
|-------|--------|----------------|
| 1 | Increase `num_workers` by 2 (2→4→…→16) | Throughput improvement < 3% |
| 2 | Increase `batch_size` by 64 (64→128→…→512) | Throughput improvement < 3% |
| 3 | Re-tune `num_workers` | Throughput improvement < 3% → **converged** |

**Safety overrides** (checked every cycle):

| Condition | Action |
|-----------|--------|
| VRAM free < 15% | Cut batch by 128 (2 steps) |
| GPU utilisation > 90% | Cut batch by 64 (1 step) |
| Throughput < 50 img/s | Skip cycle (unstable) |

A **120‑second cooldown** after any change prevents thrashing.

### Layer 3 — Loader (`loader.py`)

Wraps `torch.utils.data.DataLoader` and rebuilds it immediately when the config changes. **Deadlock prevention**:

- `multiprocessing_context='spawn'` – clean worker processes
- Forceful worker termination (`terminate()` / `kill()`) – bypasses PyTorch's graceful shutdown
- `prefetch_factor` **never changed** – avoids pipe deadlocks
- `persistent_workers=False`, `pin_memory=False`

---

## Environment

| Component | Specification |
|-----------|---------------|
| Cluster | Deucalion (University of Minho) |
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 12.4 / Driver 550.90.07 |
| PyTorch | 2.1.2 (foss-2023a-CUDA-12.1.1) |
| Python | 3.12.3 (GCCcore-13.3.0) |
| Scheduler | SLURM (partition: normal-a100-80) |
| Dataset | ImageNet ILSVRC 2012 |
| Model | AlexNet (primary) |

---

## Quick Start


### 1. Test each component

```bash
# test config
python adaptive_dataloader/config.py

# test monitor (runs for 5 seconds)
python adaptive_dataloader/monitor.py

# test controller (runs for 30 seconds)
python adaptive_dataloader/controller.py

# test loader (loads 500 images from val set)
python adaptive_dataloader/loader.py
```

### 2. Submit training jobs

```bash
# baseline — fixed config
sbatch jobs/baseline_job.sh

# adaptive — starts from bad config, adapts automatically
sbatch jobs/adaptive_job.sh

# monitor jobs
squeue -u $USER
tail -f logs/baseline_*.out
```

---

## Configuration

All parameters live in `adaptive_dataloader/config.py`:

```python
class Config:
    MODEL           = 'alexnet'      
    BATCH_SIZE      = 64            # baseline fixed batch size
    NUM_WORKERS     = 2             # baseline fixed workers
    PREFETCH_FACTOR = 1              # fixed (never adapted)
    EPOCHS          = 3
    LEARNING_RATE   = 0.05
```

Controller thresholds in `adaptive_dataloader/controller.py`:

```python
    IMPROVEMENT_THRESHOLD = 0.03      # 3% improvement threshold
    POLL_INTERVAL         = 60.0      # check every 60s
    COOLDOWN              = 120.0     # wait 120s after any change
    BATCH_STEP            = 64        # increment per batch adjustment
    WORKERS_STEP          = 2         # increment per worker adjustment
```

---


## Profiling

### torch.profiler

Both training scripts include `torch.profiler` configured to record steps 4–23:

```python
profiler = torch.profiler.profile(
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule   = schedule(wait=1, warmup=2, active=20, repeat=1),
    on_trace_ready = tensorboard_trace_handler("logs/profile_traces/"),
)
```

Open traces in Chrome: `chrome://tracing` → Load `.pt.trace.json`

### NVIDIA Nsight Systems

```bash
nsys profile \
    --output logs/profile_traces/baseline_nsys \
    python scripts/train_baseline.py
```

---



## Requirements

```
torch>=2.1.2
torchvision==0.16.2
nvidia-ml-py
flask
pandas
matplotlib
numpy
```

Install:
```bash
pip install torchvision==0.16.2 nvidia-ml-py --no-cache-dir
```

---

## Team

University of Minho · Srey Pheak , Helena , Magarida · 2026