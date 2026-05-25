# scripts/train_baseline.py
# Baseline ImageNet training
# Fixed config — no adaptation
# Includes: torch.profiler + latency + VRAM measurement

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import csv
import pathlib
import torch
import torch.nn as nn
import torch.profiler
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pynvml

from adaptive_dataloader.config import Config

# ── Setup ─────────────────────────────────────
pathlib.Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(f"logs/profile_traces/baseline_{Config.MODEL}").mkdir(
    parents=True, exist_ok=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {device}")
print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"VRAM       : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB")
print(f"Model      : {Config.MODEL}")
print(f"Batch size : {Config.BATCH_SIZE}")
print(f"Workers    : {Config.NUM_WORKERS}")
print(f"Epochs     : {Config.EPOCHS}")
print(f"LR         : {Config.LEARNING_RATE}")

# ── Data ──────────────────────────────────────
print("Loading dataset...")

train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

train_dataset = ImageFolder(Config.TRAIN_DIR, train_transform)
val_dataset   = ImageFolder(Config.VAL_DIR,   val_transform)

print(f"Train      : {len(train_dataset):,} images")
print(f"Val        : {len(val_dataset):,} images")
print(f"Classes    : {len(train_dataset.classes)}")

# no persistent_workers — same approach that worked for ResNet-50
train_loader = DataLoader(
    train_dataset,
    batch_size      = Config.BATCH_SIZE,
    num_workers     = Config.NUM_WORKERS,
    prefetch_factor = Config.PREFETCH_FACTOR,
    pin_memory      = Config.PIN_MEMORY,
    shuffle         = True,
    drop_last       = True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size  = Config.BATCH_SIZE,
    num_workers = Config.NUM_WORKERS,
    pin_memory  = Config.PIN_MEMORY,
    shuffle     = False,
)

print(f"Batches/epoch : {len(train_loader):,}")

# ── Model ─────────────────────────────────────
print(f"Loading {Config.MODEL}...")

if Config.MODEL == "resnet50":
    model = models.resnet50(weights=None)
elif Config.MODEL == "mobilenet_v2":
    model = models.mobilenet_v2(weights=None)
elif Config.MODEL == "alexnet":
    model = models.alexnet(weights=None)

model     = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr           = Config.LEARNING_RATE,
    momentum     = Config.MOMENTUM,
    weight_decay = Config.WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = Config.EPOCHS,
)

# ── GPU monitor ───────────────────────────────
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def read_gpu():
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return {
        "util"      : util.gpu,
        "vram_used" : round(mem.used  / 1024**2, 1),
        "vram_free" : round(mem.free  / 1024**2, 1),
        "vram_total": round(mem.total / 1024**2, 1),
    }

# ── Profiler ──────────────────────────────────
profiler = torch.profiler.profile(
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule = torch.profiler.schedule(
        wait   = 1,
        warmup = 2,
        active = 20,
        repeat = 1,
    ),
    on_trace_ready = torch.profiler.tensorboard_trace_handler(
        f"logs/profile_traces/baseline_{Config.MODEL}"
    ),
    record_shapes = True,
    with_stack    = False,
)

# ── CSV ───────────────────────────────────────
csv_path = f"{Config.LOG_DIR}/baseline_{Config.MODEL}_metrics.csv"
csv_file = open(csv_path, "w", newline="")
writer   = csv.DictWriter(csv_file, fieldnames=[
    "epoch",
    "train_loss",
    "train_acc",
    "val_acc",
    "throughput_img_per_sec",
    "avg_gpu_util_pct",
    "avg_gpu_idle_pct",
    "avg_vram_used_mb",
    "avg_vram_free_mb",
    "avg_load_latency_ms",
    "epoch_time_sec",
])
writer.writeheader()

# ── Training Loop ─────────────────────────────
print("Starting training...")
print("=" * 50)

with profiler:
    for epoch in range(1, Config.EPOCHS + 1):

        model.train()

        total_loss   = 0.0
        correct      = 0
        total        = 0
        gpu_utils    = []
        vram_used    = []
        vram_free    = []
        latencies_ms = []
        total_images = 0
        epoch_start  = time.perf_counter()

        for step, (images, labels) in enumerate(train_loader):
            if step >= 2000:   # ← add this line
                    break          # ← add this line
            batch_start = time.perf_counter()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            load_end   = time.perf_counter()
            latency_ms = (load_end - batch_start) * 1000
            latencies_ms.append(latency_ms)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            profiler.step()

            total_loss   += loss.item()
            total_images += images.size(0)

            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

            gpu_m = read_gpu()
            gpu_utils.append(gpu_m["util"])
            vram_used.append(gpu_m["vram_used"])
            vram_free.append(gpu_m["vram_free"])

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{Config.EPOCHS} | "
                    f"Step {step:>5} | "
                    f"Loss {total_loss/(step+1):.4f} | "
                    f"Acc {100.*correct/total:.2f}% | "
                    f"GPU {gpu_m['util']}% | "
                    f"VRAM {gpu_m['vram_used']:.0f}MB | "
                    f"Latency {latency_ms:.1f}ms"
                )

        # ── Validation ────────────────────────
        model.eval()
        val_correct = 0
        val_total   = 0

        print(f"  Running validation...")
        with torch.no_grad():
            for val_step, (images, labels) in enumerate(val_loader):
                images  = images.to(device, non_blocking=True)
                labels  = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)
                if val_step % 50 == 0:
                    print(f"  Val step {val_step}/{len(val_loader)}")

        val_acc = 100. * val_correct / val_total
        print(f"  Val Acc : {val_acc:.2f}%")

        scheduler.step()

        # ── Epoch summary ─────────────────────
        epoch_time    = time.perf_counter() - epoch_start
        avg_util      = sum(gpu_utils) / len(gpu_utils)
        avg_vram_used = sum(vram_used) / len(vram_used)
        avg_vram_free = sum(vram_free) / len(vram_free)
        avg_latency   = sum(latencies_ms) / len(latencies_ms)
        throughput    = total_images / epoch_time
        train_acc     = 100. * correct / total
        train_loss    = total_loss / len(train_loader)

        writer.writerow({
            "epoch"                  : epoch,
            "train_loss"             : round(train_loss, 4),
            "train_acc"              : round(train_acc, 2),
            "val_acc"                : round(val_acc, 2),
            "throughput_img_per_sec" : round(throughput, 2),
            "avg_gpu_util_pct"       : round(avg_util, 2),
            "avg_gpu_idle_pct"       : round(100 - avg_util, 2),
            "avg_vram_used_mb"       : round(avg_vram_used, 1),
            "avg_vram_free_mb"       : round(avg_vram_free, 1),
            "avg_load_latency_ms"    : round(avg_latency, 3),
            "epoch_time_sec"         : round(epoch_time, 2),
        })
        csv_file.flush()

        print("=" * 50)
        print(
            f"Epoch {epoch} Summary\n"
            f"  Loss        : {train_loss:.4f}\n"
            f"  Train Acc   : {train_acc:.2f}%\n"
            f"  Val Acc     : {val_acc:.2f}%\n"
            f"  Throughput  : {throughput:.0f} img/s\n"
            f"  GPU Util    : {avg_util:.1f}%\n"
            f"  Idle Time   : {100-avg_util:.1f}%\n"
            f"  VRAM Used   : {avg_vram_used:.0f} MB\n"
            f"  VRAM Free   : {avg_vram_free:.0f} MB\n"
            f"  Latency     : {avg_latency:.1f}ms\n"
            f"  Time        : {epoch_time/60:.1f} min"
        )
        print("=" * 50)

csv_file.close()
print(f"Done. Results saved → {csv_path}")
print(f"Trace saved → logs/profile_traces/baseline_{Config.MODEL}")