"""Global project parameters. Single source of truth for every module."""
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PRICES_FILE = DATA_DIR / "raw" / "prices.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DOCS_DIR = ROOT_DIR / "docs"

START_DATE = "2014-01-01"
END_DATE = "2026-10-29"

INITIAL_CAPITAL = 1_000_000
POSITION_FRACTION = 0.80
COMMISSION = 0.00125
BORROW_RATE_ANNUAL = 0.0025
TRADING_DAYS = 252

TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 6
MIN_OBSERVATIONS = 500
CORR_WINDOW = 252
CORR_THRESHOLD = 0.7
ALPHA = 0.05
JOHANSEN_DET_ORDER = 0
JOHANSEN_K_AR_DIFF = 1
HALF_LIFE_MIN = 2
HALF_LIFE_MAX = 60


def set_seed(seed=SEED):
    """Fix the random seed of every library used by the project."""
    random.seed(seed)
    np.random.seed(seed)
