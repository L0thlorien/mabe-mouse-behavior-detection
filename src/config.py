import os
from pathlib import Path

class Config:
    DATA_DIR = Path("data/mabe-mouse-behavior-detection")
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"
    TRAIN_TRACKING_DIR = DATA_DIR / "train_tracking"
    TEST_TRACKING_DIR = DATA_DIR / "test_tracking"
    TRAIN_ANNOTATION_DIR = DATA_DIR / "train_annotation"

    OUTPUT_DIR = Path("output")
    MODEL_DIR = OUTPUT_DIR / "models"
    SUBMISSION_DIR = OUTPUT_DIR / "submissions"

    MODE = "train"
    N_SPLITS = 5
    RANDOM_STATE = 42

    MODEL_TYPE = "xgboost"
    MODEL_PARAMS = {
        'n_estimators': 250,
        'learning_rate': 0.08,
        'max_depth': 6,
        'random_state': 42,
        'verbosity': 0
    }

    FPS_REFERENCE = 30.0
    TEMPORAL_WINDOWS = [5, 15, 30, 60]

    USE_CACHE = True
    CACHE_DIR = OUTPUT_DIR / "cache"
    N_JOBS = -1

    EXCLUDE_MABE22 = True
    MIN_BEHAVIOR_DURATION = 5

    SMOOTH_PREDICTIONS = True
    SMOOTH_WINDOW = 5
    MERGE_GAP_THRESHOLD = 5

    USE_AUTOENCODER = True
    AUTOENCODER_LATENT_DIM = 128
    AUTOENCODER_HIDDEN_CHANNELS = [256, 256, 256, 256]
    AUTOENCODER_SEQ_LEN = 90
    AUTOENCODER_BATCH_SIZE = 64
    AUTOENCODER_EPOCHS = 50
    AUTOENCODER_LR = 0.001
    AUTOENCODER_BETA = 0.5
    AUTOENCODER_PATIENCE = 10

    USE_PSEUDO_LABELING = True
    PSEUDO_LABEL_CONFIDENCE = 0.85
    PSEUDO_LABEL_ITERATIONS = 3

    @classmethod
    def setup_directories(cls):
        for directory in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.SUBMISSION_DIR, cls.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, section, action):
        model_dir = cls.MODEL_DIR / f"section_{section}" / action
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{cls.MODEL_TYPE}.pkl"

    @classmethod
    def get_cache_path(cls, name):
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.CACHE_DIR / f"{name}.pkl"


Config.setup_directories()
