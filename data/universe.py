"""Candidate universes, each with its economic rationale."""

UNIVERSES = {
    "bancos_eeuu": {
        "rationale": "Grandes bancos de EE.UU. con exposicion comun a tasas de interes y ciclo crediticio.",
        "tickers": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    },
    "petroleras": {
        "rationale": "Petroleras integradas cuyo factor comun es el precio del crudo.",
        "tickers": ["XOM", "CVX", "COP", "OXY"],
    },
    "utilities": {
        "rationale": "Utilities reguladas con tarifas reguladas y sensibilidad comun a tasas.",
        "tickers": ["DUK", "SO", "D", "AEP", "NEE"],
    },
    "bebidas": {
        "rationale": "Bebidas y consumo basico con demanda estable y estructuras de margen similares.",
        "tickers": ["KO", "PEP", "KDP", "MDLZ"],
    },
    "pagos": {
        "rationale": "Redes de pago expuestas al mismo volumen de consumo con tarjeta.",
        "tickers": ["V", "MA", "AXP"],
    },
    "metales": {
        "rationale": "ETFs de oro y plata y de mineras ligadas a su precio.",
        "tickers": ["GLD", "SLV", "GDX", "SIL"],
    },
}


def all_tickers():
    """Return the sorted list of every ticker across all universes."""
    return sorted({t for u in UNIVERSES.values() for t in u["tickers"]})
