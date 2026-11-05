"""Band rule on the causal z-score of the equilibrium deviation (S12 p20), with a lock after
exits that did not revert."""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG, FLAT, SHORT = 1, 0, -1


@dataclass(frozen=True)
class SignalParams:
    """Thresholds of the rule. Requires 0 <= z_out < z_in < z_stop and h_max >= 1."""

    z_in: float
    z_out: float
    z_stop: float
    h_max: int  # maximum holding period, in periods

    def __post_init__(self):
        if not 0 <= self.z_out < self.z_in < self.z_stop:
            raise ValueError("thresholds must satisfy 0 <= z_out < z_in < z_stop")
        if self.h_max < 1:
            raise ValueError("h_max must be at least 1")


def max_holding(half_life, multiple=2.0):
    """Maximum holding period as a multiple of the spread's half-life, rounded up."""
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive and finite")
    return max(1, math.ceil(multiple * half_life))


def spread_direction(z, theta):
    """Target direction of the spread in each period, from the causal z-score.

    Entry when flat: long (+1) if -z_stop < z_t < -z_in, short (-1) if z_in < z_t < z_stop.
    A long closes when z_t >= -z_out, i.e. back inside the exit band or beyond it ("reversion");
    when z_t <= -z_stop ("stop"); or after h_max periods ("h_max"). A short mirrors it.
    After a stop or an h_max exit the rule stays flat until |z_t| < z_out, so it does not reopen
    the next day on a spread that stopped reverting. A NaN z (burn-in) opens nothing.

    `theta` is a SignalParams or a dict with its fields. Returns a DataFrame with d (-1, 0, +1),
    holding (periods the open trade has been held) and exit_reason (set on the closing period).
    """
    params = theta if isinstance(theta, SignalParams) else SignalParams(**theta)
    values = z.to_numpy(dtype=float)
    d = np.zeros(len(values), dtype=int)
    holding = np.zeros(len(values), dtype=int)
    exit_reason = np.full(len(values), "", dtype=object)
    side, held, locked = FLAT, 0, False

    for t, z_t in enumerate(values):
        if side != FLAT:
            held += 1
            if side * z_t <= -params.z_stop:
                reason = "stop"
            elif side * z_t >= -params.z_out:
                reason = "reversion"
            elif held >= params.h_max:
                reason = "h_max"
            else:
                reason = ""
            holding[t] = held
            if reason:
                exit_reason[t] = reason
                locked = reason != "reversion"
                side, held = FLAT, 0
            else:
                d[t] = side
            continue

        if locked:
            locked = not abs(z_t) < params.z_out
        elif params.z_in < abs(z_t) < params.z_stop:
            side = LONG if z_t < 0 else SHORT
            d[t] = side

    return pd.DataFrame({"d": d, "holding": holding, "exit_reason": exit_reason}, index=z.index)
