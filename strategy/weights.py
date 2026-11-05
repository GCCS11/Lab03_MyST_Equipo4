"""Portfolio weights shared by the strategy modules."""
import numpy as np
import pandas as pd


def align_betas(betas, tickers, index):
    """Weights as a DataFrame with one column per ticker and one row per date.

    `betas` is either a fixed vector (Johansen) or the Filter 1 output, with columns named as
    the ticker or as `beta_<ticker>`.
    """
    if isinstance(betas, pd.DataFrame):
        frame = betas.rename(columns=lambda col: col.removeprefix("beta_"))
        missing = sorted(set(tickers) - set(frame.columns))
        if missing:
            raise ValueError(f"no weights for {missing}")
        return frame[tickers].reindex(index)
    vector = np.asarray(betas, dtype=float)
    if vector.size != len(tickers):
        raise ValueError(f"beta has {vector.size} weights but there are {len(tickers)} assets")
    return pd.DataFrame(np.tile(vector, (len(index), 1)), index=index, columns=tickers)
