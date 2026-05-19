# Adaptive DataLoader for AI Training on HPC

> A GPU-aware feedback control system that automatically adjusts PyTorch DataLoader configuration during training — eliminating the need for manual tuning.

---

## Overview

Standard PyTorch training uses a **static DataLoader configuration**. You set `batch_size`, `num_workers`, and `prefetch_factor` at the start and they never change. This creates a common problem:

- **GPU Starvation** — workers too slow → GPU idles between batches → wasted compute
- **GPU Overload** — batch too large → memory pressure → OOM errors
- **Manual Tuning** — finding optimal settings requires days of trial and error

This project solves all three by implementing a **three-layer adaptive feedback loop** that monitors GPU metrics in real time and automatically adjusts the data pipeline — without stopping training or changing the training algorithm.

---

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
│   └── profile_traces/               ← torch.profiler trace files
│
├── results/
│   └── plots/                    ← result visualizations
│
└── requirements.txt
```

---

## How It Works

### Layer 1 — GPU Monitor (`monitor.py`)

Runs in a background thread every second. Reads from `pynvml`:

- GPU utilization %
- VRAM used / free
- Throughput (images/sec)

Stores a **30-second rolling average** so the controller reads stable averages, not noisy instant values.

### Layer 2 — Adaptive Controller (`controller.py`)

Wakes up every **2 minutes** and applies four rules in priority order:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Free VRAM < 15% | Emergency: cut batch size |
| 2 | GPU util < 70% | Scale up: workers → batch → prefetch |
| 3 | GPU util > 90% | Scale down: decrease batch size |
| 4 | 70% ≤ util ≤ 90% | Do nothing — optimal zone |

A **60-second cooldown** after every change prevents thrashing.

### Layer 3 — Adaptive DataLoader (`loader.py`)

Wraps `torch.utils.data.DataLoader`. When the controller writes new values to `config.py`, the loader detects the change and **rebuilds the DataLoader automatically**. The training loop never notices.

---

## Environment

```
Cluster     : Deucalion (University of Minho)
GPU         : NVIDIA A100-SXM4-80GB
CUDA        : 12.4  /  Driver 550.90.07
PyTorch     : 2.1.2  (foss-2023a-CUDA-12.1.1)
Python      : 3.12.3  (GCCcore-13.3.0)
Scheduler   : SLURM  —  partition: normal-a100-80
Dataset     : ImageNet ILSVRC 2012
Model       : AlexNet (primary)  /  ResNet-50 (reference)
```

---

## Quick Start

### 1. Connect to cluster and load environment

```bash
srun --nodes=1 --gpus=1 --ntasks=1 \
     --cpus-per-task=8 --time=00:30:00 \
     --partition=normal-a100-80 \
     --account=<your_account> --pty bash

module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source venv_adaptive/bin/activate
cd my_advanced_computing_project
```

### 2. Test each component

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

### 3. Submit training jobs

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
    MODEL           = 'alexnet'    # alexnet / resnet50 / mobilenet_v2
    BATCH_SIZE      = 256          # baseline fixed batch size
    NUM_WORKERS     = 8            # baseline fixed workers
    PREFETCH_FACTOR = 2            # baseline fixed prefetch
    EPOCHS          = 5
    LEARNING_RATE   = 0.01
```

Controller thresholds in `adaptive_dataloader/controller.py`:

```python
UTIL_LOW       = 70.0    # below → scale up
UTIL_HIGH      = 90.0    # above → scale down
VRAM_SAFE_PCT  = 15.0    # below → emergency cut
POLL_INTERVAL  = 120.0   # seconds between evaluations
COOLDOWN       = 60.0    # seconds between adjustments
BATCH_STEP     = 32      # increment per adjustment
WORKERS_STEP   = 1       # increment per adjustment
```

---

## Baseline Results

### ResNet-50 on ImageNet (4 epochs, fixed config)

| Epoch | Loss | Train Acc | Val Acc | Throughput | GPU Util | Idle Time |
|-------|------|-----------|---------|------------|----------|-----------|
| 1 | 5.55 | 6.44% | 15.80% | 348 img/s | 91.0% | 9.0% |
| 2 | 3.74 | 24.41% | 30.24% | 363 img/s | 87.3% | 12.7% |
| 3 | 3.00 | 35.99% | 37.44% | 353 img/s | 88.9% | 11.1% |
| 4 | 2.53 | 44.46% | 47.81% | 352 img/s | 90.6% | 9.4% |

### AlexNet on ImageNet (1 epoch, fixed config)

| Epoch | Loss | Train Acc | Val Acc | Throughput | GPU Util | Idle Time |
|-------|------|-----------|---------|------------|----------|-----------|
| 1 | 6.26 | 1.67% | 3.93% | 355 img/s | 18.1% | 81.9% |

**Key finding:** AlexNet with fixed config shows 82% GPU idle time — the DataLoader cannot feed data fast enough for a fast model. This is exactly the problem the adaptive system solves.

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

## Metrics Tracked

| Metric | Source | What it tells you |
|--------|--------|-------------------|
| GPU Utilization % | pynvml | Is GPU computing or idle? |
| Data Loading Latency | timer | Gap between batches |
| Images / Second | counter | Overall pipeline speed |
| GPU Idle Time | pynvml | 100% − utilization |
| VRAM Usage | pynvml | Memory pressure |

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

University of Minho · HPC VLAB · Advanced Computing Project · 2026