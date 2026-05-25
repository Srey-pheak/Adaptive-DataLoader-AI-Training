class Config:

    TRAIN_DIR = (
        "/projects/F202500010HPCVLABUMINHO/uminhocp150/"
        "DATALOADER_PROJECT/my_advanced_computing_project/"
        "data/imagenet/train"
    )
    VAL_DIR = (
        "/projects/F202500010HPCVLABUMINHO/uminhocp150/"
        "DATALOADER_PROJECT/my_advanced_computing_project/"
        "data/imagenet/val"
    )
    LOG_DIR = "logs"

    MODEL = "alexnet"

    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    PREFETCH_FACTOR = 1
    PIN_MEMORY      = False

    EPOCHS        = 3
    LEARNING_RATE = 0.05
    MOMENTUM      = 0.9
    WEIGHT_DECAY  = 1e-4

    BATCH_MIN    = 32
    BATCH_MAX    = 512
    WORKERS_MIN  = 2
    WORKERS_MAX  = 16
    PREFETCH_MIN = 1
    PREFETCH_MAX = 1


if __name__ == "__main__":
    print("=== Config ===")
    print(f"  MODEL        : {Config.MODEL}")
    print(f"  EPOCHS       : {Config.EPOCHS}")
    print(f"  BATCH_MAX    : {Config.BATCH_MAX}")
    print(f"  PREFETCH_MAX : {Config.PREFETCH_MAX}")
    print("config.py OK")