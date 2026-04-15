# adaptive_dataloader/config.py
# All settings for the baseline training run

class Config:

    # ── Paths ─────────────────────────────────
    TRAIN_DIR = '/projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/my_advanced_computing_project/data/imagenet/train'
    VAL_DIR   = '/projects/F202500010HPCVLABUMINHO/uminhocp150/DATALOADER_PROJECT/my_advanced_computing_project/data/imagenet/val'
    LOG_DIR   = 'logs'

    # ── DataLoader ────────────────────────────
    BATCH_SIZE      = 256
    NUM_WORKERS     = 8
    PREFETCH_FACTOR = 2
    PIN_MEMORY      = True

    # ── Training ──────────────────────────────
    EPOCHS        = 5
    LEARNING_RATE = 0.1
    MOMENTUM      = 0.9
    WEIGHT_DECAY  = 1e-4


# test
if __name__ == "__main__":
    print("=== Config ===")
    print(f"BATCH_SIZE : {Config.BATCH_SIZE}")
    print(f"NUM_WORKERS: {Config.NUM_WORKERS}")
    print(f"EPOCHS     : {Config.EPOCHS}")
    print("config.py OK")