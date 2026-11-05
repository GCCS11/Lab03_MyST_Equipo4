"""Colors of the dashboard: a validated categorical palette (fixed order) plus chart chrome.

The eight categorical slots passed the CVD and contrast checks in both modes. Each asset keeps
its slot in every chart, following the ticker order of the selection, never the series rank.
"""

LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
}

DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

SPLIT_LABELS = {"train": "Entrenamiento", "test": "Prueba", "validation": "Validación"}


def palette(mode="light"):
    """Chart colors for the light or dark theme."""
    return DARK if mode == "dark" else LIGHT


def asset_colors(tickers, mode="light"):
    """Fixed color per asset, by its position in the selected group (at most 8 assets)."""
    series = palette(mode)["series"]
    if len(tickers) > len(series):
        raise ValueError("the palette has 8 slots; fold extra assets before plotting")
    return dict(zip(tickers, series))
