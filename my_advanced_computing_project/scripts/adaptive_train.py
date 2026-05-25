# scripts/adaptive_train.py
# Adaptive ImageNet training — AlexNet

if __name__ == "__main__":

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import time
    import csv
    import pathlib
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    from adaptive_dataloader.config import Config
    from adaptive_dataloader.loader import AdaptiveDataLoader

    # ── Setup ──────────────────────────────────────────────────────
    pathlib.Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}", flush=True)
    print(f"GPU     : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM    : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB", flush=True)
    print(f"Model   : {Config.MODEL}", flush=True)
    print(f"Epochs  : {Config.EPOCHS}", flush=True)
    print(f"LR      : {Config.LEARNING_RATE}", flush=True)

    # ── Adaptive starting config ───────────────────────────────────
    adaptive_config = {
        "batch_size"     : 64,
        "num_workers"    : 2,
        "prefetch_factor": 1,
    }

    print(f"\nStarting config (naive):", flush=True)
    print(f"  batch_size      : {adaptive_config['batch_size']}", flush=True)
    print(f"  num_workers     : {adaptive_config['num_workers']}", flush=True)
    print(f"  prefetch_factor : {adaptive_config['prefetch_factor']}", flush=True)
    print(flush=True)

    # ── Data ───────────────────────────────────────────────────────
    print("Loading dataset...", flush=True)

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(Config.TRAIN_DIR, train_transform)
    val_dataset   = ImageFolder(Config.VAL_DIR,   val_transform)

    print(f"Train   : {len(train_dataset):,} images", flush=True)
    print(f"Val     : {len(val_dataset):,} images", flush=True)
    print(f"Classes : {len(train_dataset.classes)}", flush=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size  = Config.BATCH_SIZE,
        num_workers = 0,
        pin_memory  = False,
        shuffle     = False,
    )

    # ── Model ──────────────────────────────────────────────────────
    print(f"Loading {Config.MODEL}...", flush=True)
    model     = models.alexnet(weights=None)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = Config.LEARNING_RATE,
        momentum     = Config.MOMENTUM,
        weight_decay = Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS,
    )

    # ── CSV ────────────────────────────────────────────────────────
    csv_path = f"{Config.LOG_DIR}/adaptive_{Config.MODEL}_metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_acc", "val_acc",
        "throughput_img_per_sec",
        "avg_gpu_util_pct",
        "avg_gpu_idle_pct",
        "avg_vram_used_mb",
        "avg_vram_free_mb",
        "epoch_time_sec",
        "final_batch_size", "final_num_workers", "final_prefetch_factor",
        "num_adjustments",
    ])
    writer.writeheader()

    # ── Training Loop ──────────────────────────────────────────────
    print("Starting adaptive training...", flush=True)
    print("=" * 50, flush=True)

    with AdaptiveDataLoader(train_dataset, adaptive_config) as loader:

        for epoch in range(1, Config.EPOCHS + 1):

            model.train()
            total_loss      = 0.0
            correct         = 0
            total           = 0
            total_images    = 0
            epoch_start     = time.perf_counter()
            adj_count_start = len(loader.controller.history)

            # FIX: track GPU util during epoch, not after
            # reading monitor after epoch end gives stale/empty rolling window
            epoch_gpu_utils  = []
            epoch_vram_used  = []
            epoch_vram_free  = []

            for step, (images, labels) in enumerate(loader):

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # gradient clipping prevents NaN loss from gradient explosion
                # especially important when batch size changes mid-training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss   += loss.item()
                total_images += images.size(0)
                _, predicted  = outputs.max(1)
                correct      += predicted.eq(labels).sum().item()
                total        += labels.size(0)

                # sample GPU metrics every 10 steps during training
                # this gives accurate per-epoch averages
                if step % 10 == 0:
                    mid_stats = loader.monitor.get_stats()
                    if mid_stats and mid_stats["n_samples"] >= 5:
                        epoch_gpu_utils.append(mid_stats["avg_util"])
                        epoch_vram_used.append(mid_stats["vram_used_mb"])
                        epoch_vram_free.append(
                            mid_stats["vram_total_mb"] - mid_stats["vram_used_mb"]
                        )

                if step % 100 == 0:
                    stats = loader.monitor.get_stats()
                    cfg   = loader.current_config
                    gpu_u = stats["avg_util"] if stats else 0.0
                    print(
                        f"Epoch {epoch}/{Config.EPOCHS} | "
                        f"Step {step:>5} | "
                        f"Loss {total_loss/(step+1):.4f} | "
                        f"Acc {100.*correct/max(total,1):.2f}% | "
                        f"GPU {gpu_u:.1f}% | "
                        f"batch={cfg['batch_size']} "
                        f"workers={cfg['num_workers']} "
                        f"prefetch={cfg['prefetch_factor']}",
                        flush=True
                    )

            # ── Validation ─────────────────────────────────────────
            model.eval()
            val_correct = 0
            val_total   = 0
            print(f"  Running validation...", flush=True)
            with torch.no_grad():
                for val_step, (images, labels) in enumerate(val_loader):
                    images  = images.to(device, non_blocking=True)
                    labels  = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total   += labels.size(0)
                    if val_step % 50 == 0:
                        print(f"  Val step {val_step}/{len(val_loader)}", flush=True)

            val_acc = 100. * val_correct / val_total
            print(f"  Val Acc : {val_acc:.2f}%", flush=True)
            scheduler.step()

            # ── Epoch metrics ───────────────────────────────────────
            epoch_time = time.perf_counter() - epoch_start
            throughput = total_images / epoch_time
            train_acc  = 100. * correct / total
            train_loss = total_loss / max(step + 1, 1)
            cfg        = loader.current_config
            n_adj      = len(loader.controller.history) - adj_count_start


            # fallback to monitor if no samples collected
            if epoch_gpu_utils:
                avg_util  = sum(epoch_gpu_utils) / len(epoch_gpu_utils)
                vram_used = sum(epoch_vram_used) / len(epoch_vram_used)
                vram_free = sum(epoch_vram_free) / len(epoch_vram_free)
            else:
                # fallback — should not happen in normal operation
                end_stats = loader.monitor.get_stats()
                avg_util  = end_stats["avg_util"]     if end_stats else 0.0
                vram_used = end_stats["vram_used_mb"] if end_stats else 0.0
                vram_free = end_stats["vram_total_mb"] - end_stats["vram_used_mb"] \
                            if end_stats else 0.0

            writer.writerow({
                "epoch"                  : epoch,
                "train_loss"             : round(train_loss, 4),
                "train_acc"              : round(train_acc, 2),
                "val_acc"                : round(val_acc, 2),
                "throughput_img_per_sec" : round(throughput, 2),
                "avg_gpu_util_pct"       : round(avg_util, 2),
                "avg_gpu_idle_pct"       : round(100 - avg_util, 2),
                "avg_vram_used_mb"       : round(vram_used, 1),
                "avg_vram_free_mb"       : round(vram_free, 1),
                "epoch_time_sec"         : round(epoch_time, 2),
                "final_batch_size"       : cfg["batch_size"],
                "final_num_workers"      : cfg["num_workers"],
                "final_prefetch_factor"  : cfg["prefetch_factor"],
                "num_adjustments"        : n_adj,
            })
            csv_file.flush()

            print("=" * 50, flush=True)
            print(
                f"Epoch {epoch} Summary\n"
                f"  Loss          : {train_loss:.4f}\n"
                f"  Train Acc     : {train_acc:.2f}%\n"
                f"  Val Acc       : {val_acc:.2f}%\n"
                f"  Throughput    : {throughput:.0f} img/s\n"
                f"  GPU Util      : {avg_util:.1f}%\n"
                f"  Idle Time     : {100-avg_util:.1f}%\n"
                f"  VRAM Used     : {vram_used:.0f} MB\n"
                f"  VRAM Free     : {vram_free:.0f} MB\n"
                f"  Batch size    : {cfg['batch_size']}\n"
                f"  Workers       : {cfg['num_workers']}\n"
                f"  Prefetch      : {cfg['prefetch_factor']}\n"
                f"  Adjustments   : {n_adj}\n"
                f"  Time          : {epoch_time/60:.1f} min",
                flush=True
            )
            print("=" * 50, flush=True)

    # ── Adjustment history ─────────────────────────────────────────
    print("\nAll adjustments made during training:", flush=True)
    if loader.controller.history:
        for entry in loader.controller.history:
            t = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
            print(
                f"  [{t}] {entry['reason']} "
                f"(GPU {entry['gpu_util']}% | VRAM free {entry['vram_free']}%)",
                flush=True
            )
    else:
        print("  None", flush=True)

    csv_file.close()
    print(f"\nDone. Results saved → {csv_path}", flush=True)