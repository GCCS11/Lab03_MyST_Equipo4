"""Figuras 7 y 8 del reporte: la curva de capital con su drawdown y la distribucion de
retornos por operacion. Leen los artifacts de la corrida unica de main.py (equity.csv y
operaciones.csv) y nunca vuelven a correr el backtest.

Correr con: python -m performance.figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import ARTIFACTS_DIR, DOCS_DIR
from dashboard.theme import palette

FIGURES_DIR = DOCS_DIR / "figures"
SPLIT_NAMES = (("train", "Entrenamiento"), ("test", "Prueba"), ("validation", "Validación"))


def load_equity(directory=ARTIFACTS_DIR):
    return pd.read_csv(directory / "equity.csv", index_col="date", parse_dates=True)


def load_operaciones(directory=ARTIFACTS_DIR):
    return pd.read_csv(directory / "operaciones.csv", parse_dates=["entrada", "salida"])


def _mark_splits(axis, frame, colors):
    """Una linea vertical en cada corte de particion presente, y el nombre de cada tramo
    (Entrenamiento, Prueba, Validacion) escrito sobre la grafica, en el orden en que
    arrancan (no el de SPLIT_NAMES, por si falta alguna particion)."""
    starts = frame.reset_index().groupby("split", sort=False)["date"].min().sort_values()
    for position, (split, start) in enumerate(starts.items()):
        name = dict(SPLIT_NAMES).get(split, split)
        if position > 0:
            axis.axvline(start, color=colors["muted"], linewidth=1)
        axis.text(
            start, 0.97, f" {name}", transform=axis.get_xaxis_transform(),
            ha="left", va="top", fontsize=9, color=colors["ink_secondary"],
        )


def fig7_equity(equity, mode="light"):
    """Figura 7: la curva de capital es un solo backtest continuo de train a validacion (no
    reinicia en cada particion); las metricas si se calculan por separado en cada tramo, que es
    lo que rotula esta grafica. El drawdown va superpuesto en un eje secundario."""
    colors = palette(mode)
    figure, ax_equity = plt.subplots(figsize=(10, 4.8))

    ax_equity.plot(
        equity.index, equity["equity"], color=colors["series"][0], linewidth=1.4, label="Capital",
    )
    ax_equity.set_xlabel("Fecha")
    ax_equity.set_ylabel("Capital (USD)")

    ax_drawdown = ax_equity.twinx()
    ax_drawdown.fill_between(
        equity.index, equity["drawdown"], 0, color=colors["series"][7], alpha=0.18, linewidth=0,
    )
    ax_drawdown.plot(
        equity.index, equity["drawdown"], color=colors["series"][7], linewidth=1.0, label="Drawdown",
    )
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.set_ylim(min(-0.01, float(equity["drawdown"].min()) * 1.2), 0.02)

    _mark_splits(ax_equity, equity, colors)

    lines_equity, labels_equity = ax_equity.get_legend_handles_labels()
    lines_drawdown, labels_drawdown = ax_drawdown.get_legend_handles_labels()
    ax_equity.legend(
        lines_equity + lines_drawdown, labels_equity + labels_drawdown,
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False,
    )

    ax_equity.set_title("Figura 7. Capital continuo y drawdown, con metricas por particion")
    return figure


def fig8_distribucion(operaciones, mode="light"):
    """Figura 8: histograma de retornos por operacion con la normal ajustada superpuesta, y un
    texto con media, desviacion, asimetria y n. Sin operaciones, solo deja un aviso."""
    colors = palette(mode)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    returns = operaciones["retorno"].dropna().to_numpy(dtype=float) * 100.0 if len(operaciones) else np.array([])

    if len(returns) >= 2:
        bins = min(40, max(8, int(np.sqrt(len(returns))) * 2))
        _, edges, _ = axis.hist(
            returns, bins=bins, color=colors["series"][0], edgecolor=colors["surface"],
            label="Operaciones",
        )
        mean, std, skew = returns.mean(), returns.std(ddof=1), float(pd.Series(returns).skew())
        grid = np.linspace(edges[0], edges[-1], 200)
        width = edges[1] - edges[0]
        axis.plot(
            grid, len(returns) * width * stats.norm.pdf(grid, mean, std),
            color=colors["ink"], linewidth=2, label="Normal ajustada",
        )
        axis.axvline(mean, color=colors["muted"], linestyle="--", linewidth=1)
        texto = f"media = {mean:.2f}%, desv = {std:.2f}%, asimetría = {skew:.2f}, n = {len(returns)}"
        axis.text(
            0.02, 0.95, texto, transform=axis.transAxes, ha="left", va="top", fontsize=9,
            color=colors["ink_secondary"],
        )
        axis.legend(fontsize=9)
    else:
        axis.text(
            0.5, 0.5, "No hay operaciones suficientes", ha="center", va="center",
            transform=axis.transAxes, fontsize=11, color=colors["muted"],
        )

    axis.set_title("Figura 8. Distribución de retornos por operación")
    axis.set_xlabel("Retorno neto por operación (%)")
    axis.set_ylabel("Operaciones")
    figure.tight_layout()
    return figure


def main(directory=ARTIFACTS_DIR, out_dir=FIGURES_DIR):
    equity = load_equity(directory)
    operaciones = load_operaciones(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure7 = fig7_equity(equity)
    figure7.savefig(out_dir / "fig7_equity.png", dpi=160, bbox_inches="tight")
    plt.close(figure7)

    figure8 = fig8_distribucion(operaciones)
    figure8.savefig(out_dir / "fig8_retornos.png", dpi=160)
    plt.close(figure8)


if __name__ == "__main__":
    main()
