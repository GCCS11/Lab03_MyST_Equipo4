"""Static copy of the dashboard in docs/dashboard.html, readable without Streamlit.

Run from the repo root: python -m dashboard.export
"""
from pathlib import Path

from dashboard import figures
from dashboard.data import artifacts_dir, load_artifacts
from dashboard.presentation import build_steps, load_extras
from dashboard.theme import SPLIT_LABELS

DOCS_FILE = Path(__file__).resolve().parents[1] / "docs" / "dashboard.html"


def export_html(artifacts, path=DOCS_FILE, mode="light", extras=None):
    """Write every chart of both views and the metrics table to one self-contained HTML file.

    With `extras` (presentation.load_extras) it also writes the 10 steps of the presentation.
    """
    splits = list(SPLIT_LABELS)
    sections = ["<h2>Vista ejecutiva</h2>"]
    if artifacts["metricas"]:
        sections.append(figures.metrics_table(artifacts["metricas"]).to_html(classes="metrics", border=0))
    panels = figures.executive_panels(artifacts, splits, mode)
    panels_technical = figures.technical_panels(artifacts, splits, mode)
    include_js = True
    for title, group in (("", panels), ("<h2>Vista técnica</h2>", panels_technical)):
        sections.append(title)
        for panel in group:
            if panel.figure is None:
                sections.append(f"<p>Falta {panel.source}.</p>")
                continue
            sections.append(panel.figure.to_html(full_html=False, include_plotlyjs=include_js))
            include_js = False

    if extras is not None:
        sections.append("<h2>Presentación</h2>")
        for number, step in enumerate(build_steps(artifacts, extras, mode), start=1):
            sections.append(f"<h3>{number}. {step['titulo']} <small>(expone {step['quien']})</small></h3>")
            sections.append("<p>" + " · ".join(f"<b>{label}:</b> {value}" for label, value, _ in step["cifras"]) + "</p>")
            if step["figura"] is not None:
                sections.append(step["figura"].to_html(full_html=False, include_plotlyjs=include_js))
                include_js = False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Lab 03 · Dashboard</title><style>"
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;"
        "color:#0b0b0b;max-width:1200px;margin:0 auto;padding:16px}"
        "table.metrics{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        "table.metrics td,table.metrics th{padding:4px 12px;border-bottom:1px solid #e1e0d9;text-align:right}"
        "</style></head><body><h1>Lab 03 · Arbitraje estadístico</h1>"
        "<p>Copia estática del dashboard (streamlit run dashboard/app.py), generada desde artifacts/.</p>"
        + "".join(sections) + "</body></html>",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    print(f"Escrito {export_html(load_artifacts(), extras=load_extras(artifacts_dir()))}")
