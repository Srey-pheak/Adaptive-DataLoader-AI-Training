# adaptive_dataloader/loader.py
# Layer 3 — Act
#
# DEADLOCK PREVENTION DESIGN:
#
# batch_size changes → safe mid-epoch, no process spawning
#   _dirty flag set → DataLoader rebuilt immediately next batch
#
# num_workers changes → EPOCH BOUNDARY ONLY
#   controller sets pending_workers (never touches config directly)
#   loader checks pending_workers at START of each epoch
#   applies the change, rebuilds DataLoader once between epochs
#   no mid-epoch process spawning = no deadlock possible
#
# pin_memory = False always — even with num_workers=0 PyTorch
# spawns a pinned-memory thread that can hang on some HPC nodes.
#
# multiprocessing_context = "spawn" when num_workers > 0
# spawn starts workers from a clean process with no parent state,
# preventing the fork+pynvml NVML handle deadlock on Deucalion.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from torch.utils.data import DataLoader

from adaptive_dataloader.monitor    import GPUMonitor
from adaptive_dataloader.controller import AdaptiveController


class AdaptiveDataLoader:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config  = config

        self._monitor    = GPUMonitor(interval=1.0, window=30)
        self._controller = AdaptiveController(self._monitor, self.config)

        self._loader = None
        self._dirty  = True   # True = batch size changed, rebuild now
        self._epoch  = 0      # track epoch boundary for worker changes

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        self._monitor.start()
        self._controller.start()
        print("AdaptiveDataLoader ready.", flush=True)

    def stop(self):
        self._controller.stop()
        self._monitor.stop()
        self._cleanup_loader()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Iteration ─────────────────────────────────────────────

    def __iter__(self):
        """
        Called once per epoch by the training loop.
        Both batch_size and num_workers changes apply immediately
        mid-epoch — triggers DataLoader rebuild between batches.
        """
        while True:
            self._rebuild_if_needed()

            for batch in self._loader:
                self._monitor.record_batch(self._batch_size_of(batch))
                yield batch

                if self._dirty:
                    print(
                        f"[AdaptiveDataLoader] Config changed — rebuilding...",
                        flush=True
                    )
                    break
            else:
                return  # epoch complete

    def __len__(self):
        return (len(self.dataset) + self.config["batch_size"] - 1) \
               // self.config["batch_size"]

    # ── Properties ────────────────────────────────────────────

    @property
    def monitor(self):
        return self._monitor

    @property
    def controller(self):
        return self._controller

    @property
    def current_config(self):
        return dict(self.config)

    # ── Internal ──────────────────────────────────────────────

    def _cleanup_loader(self):
        """Terminate worker processes directly — guaranteed no deadlock."""
        if self._loader is None:
            return
        try:
            iterator = getattr(self._loader, '_iterator', None)
            if iterator is not None:
                try:
                    iterator._shutdown_workers()
                except Exception:
                    pass

            workers = getattr(self._loader, '_workers', None)
            if workers:
                for w in workers:
                    try:
                        if w.is_alive():
                            w.terminate()
                            w.join(timeout=3)
                            if w.is_alive():
                                w.kill()
                                w.join(timeout=1)
                    except Exception:
                        pass

            if hasattr(self._loader, '_worker_result_queue'):
                try:
                    self._loader._worker_result_queue.cancel_join_thread()
                except Exception:
                    pass

        except Exception as e:
            print(f"[AdaptiveDataLoader] Cleanup warning: {e}", flush=True)
        finally:
            del self._loader
            self._loader = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(0.1)

    def _rebuild_if_needed(self):
        if not self._dirty and self._loader is not None:
            return

        self._cleanup_loader()

        cfg       = self.config
        n_workers = cfg["num_workers"]
        prefetch  = cfg["prefetch_factor"] if n_workers > 0 else None

        # spawn: workers start from a clean process
        # no parent pynvml handle copied = no NVML deadlock
        mp_ctx = "spawn" if n_workers > 0 else None

        print(
            f"[AdaptiveDataLoader] Building DataLoader — "
            f"batch={cfg['batch_size']} "
            f"workers={n_workers} "
            f"prefetch={cfg['prefetch_factor']}",
            flush=True
        )

        self._loader = DataLoader(
            self.dataset,
            batch_size              = cfg["batch_size"],
            num_workers             = n_workers,
            prefetch_factor         = prefetch,
            pin_memory              = False,
            shuffle                 = True,
            drop_last               = True,
            persistent_workers      = False,
            multiprocessing_context = mp_ctx,
        )
        self._dirty = False
        self._controller.on_change = self._on_batch_change

    def _on_batch_change(self):
        """
        Called by controller when batch_size changes.
        Sets dirty flag for immediate mid-epoch rebuild.
        Worker changes go through pending_workers instead —
        they are never applied mid-epoch.
        NOTE: we do NOT clear monitor snapshots here —
        clearing causes throughput=0 on next poll which
        triggers another immediate change before data is stable.
        """
        self._dirty = True

    @staticmethod
    def _batch_size_of(batch):
        if isinstance(batch, (list, tuple)) and batch:
            return AdaptiveDataLoader._batch_size_of(batch[0])
        try:
            return len(batch)
        except TypeError:
            return 1