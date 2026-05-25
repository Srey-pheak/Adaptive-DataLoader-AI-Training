# adaptive_dataloader/monitor.py
# Layer 1 — Measure

import time
import threading
import collections


class GPUSnapshot:
    def __init__(self, gpu_util, vram_used_mb, vram_total_mb, throughput):
        self.timestamp     = time.time()
        self.gpu_util      = gpu_util
        self.vram_used_mb  = vram_used_mb
        self.vram_total_mb = vram_total_mb
        self.throughput    = throughput

    @property
    def vram_free_pct(self):
        return 100.0 * (1.0 - self.vram_used_mb / max(self.vram_total_mb, 1))

    def __str__(self):
        return (
            f"GPU={self.gpu_util:.1f}% "
            f"VRAM={self.vram_used_mb:.0f}MB "
            f"Free={self.vram_free_pct:.1f}% "
            f"Throughput={self.throughput:.1f}img/s"
        )


class GPUMonitor:
    """
    Runs in a background thread.
    Reads GPU metrics every second.
    Stores last 30 readings for averaging.

    pynvml is imported lazily inside start() — NOT at module level.
    Importing pynvml at module level triggers NVML driver initialisation
    on some HPC nodes (Deucalion) and hangs before any user code runs.
    """

    def __init__(self, device_index=0, interval=1.0, window=30):
        self.device_index = device_index
        self.interval     = interval
        self.window       = window

        self._snapshots = collections.deque(maxlen=window)
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._handle    = None

        self._images_since_last = 0
        self._last_time         = time.monotonic()
        self._throughput        = 0.0

        # pynvml is NOT imported here — deferred to start()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self):
        # import pynvml here, not at module level
        # this avoids hanging on clusters where NVML init
        # blocks during Python import phase
        import pynvml
        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        self._running = True
        self._thread  = threading.Thread(
            target=self._loop,
            daemon=True,
            name="GPUMonitor"
        )
        self._thread.start()
        print("GPUMonitor started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        print("GPUMonitor stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Called by training loop ────────────────────────────────

    def record_batch(self, n_images):
        with self._lock:
            self._images_since_last += n_images

    # ── Called by controller ───────────────────────────────────

    def get_stats(self):
        with self._lock:
            snaps = list(self._snapshots)

        if not snaps:
            return None

        avg_util = sum(s.gpu_util   for s in snaps) / len(snaps)
        avg_tp   = sum(s.throughput for s in snaps) / len(snaps)
        last     = snaps[-1]

        return {
            "avg_util"      : round(avg_util, 2),
            "avg_throughput": round(avg_tp, 2),
            "vram_used_mb"  : round(last.vram_used_mb, 1),
            "vram_total_mb" : round(last.vram_total_mb, 1),
            "vram_free_pct" : round(last.vram_free_pct, 1),
            "n_samples"     : len(snaps),
        }

    def get_latest(self):
        with self._lock:
            if self._snapshots:
                return self._snapshots[-1]
            return None

    # ── Internal ───────────────────────────────────────────────

    def _loop(self):
        while self._running:
            snap = self._sample()
            with self._lock:
                self._snapshots.append(snap)
            time.sleep(self.interval)

    def _sample(self):
        now = time.monotonic()

        with self._lock:
            elapsed                 = now - self._last_time
            images                  = self._images_since_last
            self._images_since_last = 0
            self._last_time         = now

        if elapsed > 0:
            self._throughput = images / elapsed
        throughput = self._throughput

        # pynvml calls outside the lock — slow kernel calls
        # should never block record_batch() in the training loop
        util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem  = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)

        return GPUSnapshot(
            gpu_util      = float(util.gpu),
            vram_used_mb  = mem.used  / 1024**2,
            vram_total_mb = mem.total / 1024**2,
            throughput    = throughput,
        )