"""Figures 2 and 6 of the report: the portfolio spread with the trading signals, and the causal
z-score with the frozen thresholds. Both read the artifacts of the single run of main.py.

Run with: python -m strategy.figures
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import ARTIFACTS_DIR, DOCS_DIR
from dashboard.theme import palette

FIGURES_DIR = DOCS_DIR / "figures"
SPLIT_NAMES = (("train", "Entrenamiento"), ("test", "Prueba"), ("validation", "Validación"))


def load_signals(directory=ARTIFACTS_DIR):
    """Signals from the first valid z-score on (the filters' burn-in is left out)."""
    signals = pd.read_csv(directory / "senales.csv", index_col="date", parse_dates=True)
    return signals.loc[signals["z"].first_valid_index():]


def mark_splits(axis, signals, colors):
    """A vertical line at each cut date and the name of each partition above its stretch."""
    starts = signals.reset_index().groupby("split")["date"].min()
    for position, (split, name) in enumerate(SPLIT_NAMES):
        if split not in starts:
            continue
        if position > 0:
            axis.axvline(starts[split], color=colors["muted"], linewidth=1)
        axis.text(
            starts[split], 0.98, f" {name}", transform=axis.get_xaxis_transform(),
            ha="left", va="top", fontsize=9, color=colors["ink_secondary"],
        )


def finish(figure, axis, title, y_label, path):
    axis.set_title(title)
    axis.set_xlabel("Fecha")
    axis.set_ylabel(y_label)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=9, frameon=False)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_spread(signals, path, mode="light"):
    """Figure 2: equilibrium deviation S_t, its latent mean from Filter 2, and the day each
    position was decided (it executes at the next close) and closed."""
    colors = palette(mode)
    d = signals["d"].astype(int)
    previous = d.shift(fill_value=0)
    events = (
        ("Entrada larga", (d == 1) & (previous != 1), "^", colors["series"][2]),
        ("Entrada corta", (d == -1) & (previous != -1), "v", colors["series"][1]),
        ("Salida", signals["exit_reason"].fillna("") != "", "x", colors["ink"]),
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(signals.index, signals["spread"], color=colors["series"][0], linewidth=1, label="Spread S_t")
    axis.plot(signals.index, signals["mu_prior"], color=colors["muted"], linewidth=1.5, label="Media estimada (Filtro 2)")
    for name, mask, marker, color in events:
        axis.scatter(
            signals.index[mask], signals["spread"][mask], marker=marker, s=36, color=color,
            edgecolors=None if marker == "x" else colors["surface"], linewidths=0.5 if marker != "x" else 1.2,
            zorder=3, label=f"{name} ({int(mask.sum())})",
        )
    mark_splits(axis, signals, colors)
    finish(figure, axis, "Figura 2. Spread del portafolio con las señales de entrada y salida",
           "Desviación de equilibrio (log)", path)


def plot_zscore(signals, theta, path, mode="light"):
    """Figure 6: causal z-score with the frozen entry, exit and stop thresholds."""
    colors = palette(mode)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(signals.index, signals["z"], color=colors["series"][0], linewidth=0.9, label="z_t")
    axis.axhline(0, color=colors["baseline"], linewidth=0.8)
    thresholds = (
        ("z_stop", theta["z_stop"], "-", colors["series"][7]),
        ("z_in", theta["z_in"], "--", colors["ink"]),
        ("z_out", theta["z_out"], ":", colors["ink"]),
    )
    for name, level, style, color in thresholds:
        axis.axhline(level, color=color, linestyle=style, linewidth=1, label=f"±{name} = {level:g}")
        axis.axhline(-level, color=color, linestyle=style, linewidth=1)
    mark_splits(axis, signals, colors)
    finish(figure, axis, "Figura 6. z-score causal del Filtro 2 con los umbrales congelados", "z-score", path)


def main(directory=ARTIFACTS_DIR, out_dir=FIGURES_DIR):
    signals = load_signals(directory)
    theta = json.loads((directory / "theta.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_spread(signals, out_dir / "fig2_spread.png")
    plot_zscore(signals, theta, out_dir / "fig6_zscore.png")


if __name__ == "__main__":
    main()
