# adaptive_dataloader/controller.py
# Layer 2 — Decide
#
# ALGORITHM: Coordinate descent with throughput improvement detection
#
# Phase 1 — scale workers 2->16, step 2, stop at <3% improvement
# Phase 2 — scale batch  64->512, step 64, stop at <3% improvement
# Phase 3 — re-tune workers, converge when <3%
#
# Workers scaled MID-EPOCH via SIGTERM/SIGKILL (loader handles this)
# Batch scaled MID-EPOCH (always safe)
# Prefetch NEVER scaled — causes pipe deadlock

import time
import threading


class AdaptiveController:

    def __init__(self, monitor, config):
        self.monitor = monitor
        self.config  = config

        # ── Convergence ──────────────────────────────────────────
        self.IMPROVEMENT_THRESHOLD = 0.03   # 3% minimum to continue
        self.MIN_THROUGHPUT        = 50.0   # ignore readings below this

        # ── Safety ───────────────────────────────────────────────
        self.VRAM_SAFE_PCT = 15.0
        self.UTIL_HIGH     = 90.0

        # ── Batch bounds ─────────────────────────────────────────
        self.BATCH_MIN  = 64
        self.BATCH_MAX  = 512
        self.BATCH_STEP = 64

        # ── Worker bounds ────────────────────────────────────────
        self.WORKERS_MIN  = 2
        self.WORKERS_MAX  = 16
        self.WORKERS_STEP = 2

        # ── Timing ───────────────────────────────────────────────
        self.POLL_INTERVAL = 60.0
        self.COOLDOWN      = 150.0   # 2.5 min after any change

        # ── State ────────────────────────────────────────────────
        self._running         = False
        self._thread          = None
        self._last_adjustment = time.monotonic() + 120.0  # 2 min warmup
        self._lock            = threading.Lock()
        self.on_change        = None
        self.history          = []
        self._converged       = False

        # ── Coordinate descent state ─────────────────────────────
        self._phase     = 1
        self._tp_before = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop,
            daemon=True,
            name="AdaptiveController"
        )
        self._thread.start()
        print("AdaptiveController started.", flush=True)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        print("AdaptiveController stopped.", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Internal loop ─────────────────────────────────────────────

    def _loop(self):
        while self._running:
            time.sleep(self.POLL_INTERVAL)
            self._evaluate()

    def _evaluate(self):

        if self._converged:
            return

        now = time.monotonic()
        if now - self._last_adjustment < self.COOLDOWN:
            return

        stats = self.monitor.get_stats()
        if stats is None:
            return
        if stats["n_samples"] < 15:
            print("[Controller] Waiting for stable data (need 15 samples)", flush=True)
            return

        current_tp = stats["avg_throughput"]
        util       = stats["avg_util"]
        vram_free  = stats["vram_free_pct"]

        with self._lock:
            batch   = self.config["batch_size"]
            workers = self.config["num_workers"]

        print(
            f"[Controller] Phase {self._phase} | "
            f"throughput={current_tp:.0f} img/s | "
            f"GPU={util:.1f}% | "
            f"batch={batch} workers={workers}",
            flush=True
        )

        if current_tp < self.MIN_THROUGHPUT:
            return

        # ── Safety: VRAM critical ─────────────────────────────────
        if vram_free < self.VRAM_SAFE_PCT:
            new_batch = max(self.BATCH_MIN, batch - self.BATCH_STEP * 2)
            with self._lock:
                self.config["batch_size"] = new_batch
            self._tp_before = 0.0
            self._record(
                f"VRAM critical ({vram_free:.1f}% free) -> batch {batch}->{new_batch}",
                now, util, vram_free
            )
            return

        # ── Safety: GPU overloaded ────────────────────────────────
        if util > self.UTIL_HIGH:
            new_batch = max(self.BATCH_MIN, batch - self.BATCH_STEP)
            with self._lock:
                self.config["batch_size"] = new_batch
            self._tp_before = 0.0
            self._record(
                f"GPU overloaded ({util:.1f}%) -> batch {batch}->{new_batch}",
                now, util, vram_free
            )
            return

        # ── Check improvement from last change ────────────────────
        if self._tp_before > 0:
            improvement = (current_tp - self._tp_before) / self._tp_before
            print(
                f"[Controller] Improvement: "
                f"{self._tp_before:.0f} -> {current_tp:.0f} img/s "
                f"({improvement*100:+.1f}%)",
                flush=True
            )

            if improvement < self.IMPROVEMENT_THRESHOLD:
                # this phase is exhausted — advance
                self._advance_phase(current_tp, util, vram_free, now, batch, workers)
                return

        # ── Continue current phase ────────────────────────────────
        self._tp_before = current_tp
        self._act(now, util, vram_free, batch, workers)

    def _act(self, now, util, vram_free, batch, workers):
        """Scale the parameter for the current phase.
        NOTE: _record() called OUTSIDE self._lock to avoid reentrant deadlock.
        """
        if self._phase == 1 or self._phase == 3:
            # scale workers mid-epoch — loader handles SIGTERM/SIGKILL rebuild
            workers_maxed = False
            new_w = None
            with self._lock:
                if self.config["num_workers"] < self.WORKERS_MAX:
                    new_w = self.config["num_workers"] + self.WORKERS_STEP
                    self.config["num_workers"] = new_w
                else:
                    workers_maxed = True

            if workers_maxed:
                self._advance_phase(self._tp_before, util, vram_free, now, batch, workers)
            else:
                # _record OUTSIDE lock
                self._record(
                    f"Phase {self._phase} workers: {workers}->{new_w}",
                    now, util, vram_free
                )

        elif self._phase == 2:
            # scale batch — always safe mid-epoch
            batch_maxed = False
            reason = None
            with self._lock:
                if self.config["batch_size"] < self.BATCH_MAX:
                    old_b = self.config["batch_size"]
                    self.config["batch_size"] = min(
                        self.BATCH_MAX,
                        self.config["batch_size"] + self.BATCH_STEP
                    )
                    reason = f"Phase 2 batch: {old_b}->{self.config['batch_size']}"
                else:
                    batch_maxed = True

            if batch_maxed:
                self._advance_phase(self._tp_before, util, vram_free, now, batch, workers)
            else:
                # _record OUTSIDE lock
                self._record(reason, now, util, vram_free)

    def _advance_phase(self, current_tp, util, vram_free, now, batch, workers):
        """Move to next phase or converge.

        CRITICAL: reset _tp_before to 0.0 on EVERY phase transition.
        If we carry over _tp_before from the previous phase, the first
        poll in the new phase compares against old-phase throughput and
        may immediately declare the new phase exhausted without ever
        making a single change. Setting to 0.0 guarantees the first
        poll in each new phase always acts before checking improvement.
        """
        if self._phase == 1:
            print(
                f"[Controller] Phase 1 complete — workers={workers} "
                f"at {current_tp:.0f} img/s. Moving to phase 2 (batch).",
                flush=True
            )
            self._phase     = 2
            self._tp_before = 0.0   # CRITICAL: reset, do not carry over

        elif self._phase == 2:
            print(
                f"[Controller] Phase 2 complete — batch={batch} "
                f"at {current_tp:.0f} img/s. Moving to phase 3 (re-tune workers).",
                flush=True
            )
            self._phase     = 3
            self._tp_before = 0.0   # CRITICAL: reset, do not carry over

        elif self._phase == 3:
            self._converged = True
            with self._lock:
                fb = self.config["batch_size"]
                fw = self.config["num_workers"]
            print(
                f"[Controller] CONVERGED\n"
                f"  throughput  : {current_tp:.0f} img/s\n"
                f"  final config: batch={fb} workers={fw}\n"
                f"  improvement threshold {self.IMPROVEMENT_THRESHOLD*100:.0f}% "
                f"not met in any phase",
                flush=True
            )

    def _record(self, reason, now, util, vram_free):
        """Record adjustment — always called OUTSIDE self._lock."""
        self._last_adjustment = now
        with self._lock:
            entry = {
                "time"       : time.time(),
                "reason"     : reason,
                "gpu_util"   : round(util, 1),
                "vram_free"  : round(vram_free, 1),
                "batch_size" : self.config["batch_size"],
                "num_workers": self.config["num_workers"],
                "prefetch"   : self.config["prefetch_factor"],
            }
        self.history.append(entry)
        print(f"[Controller] {reason}", flush=True)
        if self.on_change:
            self.on_change()