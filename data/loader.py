"""Price ingestion, validation and chronological partitioning."""
import pandas as pd
import yfinance as yf

from config import (
    END_DATE,
    PRICES_FILE,
    START_DATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
)
from data.universe import all_tickers


def download_prices(tickers, start=START_DATE, end=END_DATE):
    """Download adjusted daily closes for the given tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index().dropna(how="all")
    return prices[sorted(prices.columns)]


def load_prices(refresh=False):
    """Load prices from the local snapshot, downloading it if missing."""
    if PRICES_FILE.exists() and not refresh:
        return pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    prices = download_prices(all_tickers())
    PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_FILE)
    return prices


def split_positions(n_rows):
    """Return the row positions where train and test end."""
    train_end = int(n_rows * TRAIN_FRACTION)
    test_end = int(n_rows * (TRAIN_FRACTION + TEST_FRACTION))
    return train_end, test_end


def partition_prices(prices):
    """Split prices chronologically into train, test and validation."""
    train_end, test_end = split_positions(len(prices))
    return {
        "train": prices.iloc[:train_end],
        "test": prices.iloc[train_end:test_end],
        "validation": prices.iloc[test_end:],
    }


def load_train_prices():
    """Load only the training partition. Screening must use this function."""
    return partition_prices(load_prices())["train"]


def validate_prices(prices):
    """Summarize coverage and missing values per ticker."""
    return pd.DataFrame(
        {
            "first_date": prices.apply(lambda s: s.first_valid_index()),
            "last_date": prices.apply(lambda s: s.last_valid_index()),
            "n_obs": prices.count(),
            "n_missing": prices.isna().sum(),
        }
    )
