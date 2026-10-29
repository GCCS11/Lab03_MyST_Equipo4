"""Global project parameters. Single source of truth for every module."""
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DOCS_DIR = ROOT_DIR / "docs"

INITIAL_CAPITAL = 1_000_000
POSITION_FRACTION = 0.80
COMMISSION = 0.00125
BORROW_RATE_ANNUAL = 0.0025
TRADING_DAYS = 252

TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20


def set_seed(seed=SEED):
    """Fix the random seed of every library used by the project."""
    random.seed(seed)
    np.random.seed(seed)
