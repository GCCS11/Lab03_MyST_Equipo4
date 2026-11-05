# Lab03_MyST_Equipo4: arbitraje estadístico con cointegración y filtros de Kalman

Microestructura y Sistemas de Trading, ITESO, otoño 2026. Laboratorio 03.

> Esqueleto del 4-nov. Las partes marcadas **PENDIENTE** se completan conforme se fusionen los módulos.

## Integrantes
| Integrante | GitHub | Responsabilidad |
|---|---|---|
| Juan Pablo Barba González | JPBAG0 | Estrategia (VECM, señales, tamaño de posición), `main.py`, dashboard y gestión del proyecto |
| Gian Carlo Campos Sayavedra | GCCS11 | Datos, selección del grupo (cointegración), backtest y métricas |
| Ian Carlo Escalante Martínez | ianmartinez01 | Filtros de Kalman, formulación de Powell y reporte |

## Nivel de alcance
**Un grupo de 6 activos** (2 × n, con n = 3 integrantes), **sin Risk Parity**. No se implementan la sección 3.7, la figura 9 ni la parte de nivel B de la prueba 4. PENDIENTE: confirmar con el profesor el nombre del nivel.

## Descripción
Estrategia de arbitraje estadístico sobre un portafolio de 6 utilities eléctricas reguladas de EE. UU., con datos diarios de 2014 a 2026. El flujo es:
1. **Selección del grupo.** Se aplica el embudo del 3.2 solo sobre entrenamiento: universo con justificación económica, correlación móvil de retornos, Engle-Granger, Johansen, validación del spread y estabilidad. Se registran todos los grupos evaluados.
2. **Filtro 1 (Kalman).** Estima los pesos dinámicos β̂_t del portafolio reversivo, con la pata normalizada en +1.
3. **VECM.** Se estima con el vector de Johansen como término de corrección de error. Da la velocidad de ajuste βᵀα y la vida media que calibra la tenencia máxima.
4. **Filtro 2 (Kalman).** Estima la media y la varianza latentes de la desviación de equilibrio. El z-score usa los estadísticos previos a la actualización.
5. **Regla y tamaño de posición.** Se entra si z_in < abs(z) < z_stop y se sale si abs(z) < z_out, si se activa el stop o si se llega a la tenencia máxima. Se usa 80% del capital, repartido en proporción a abs(β̂_t).
6. **Backtest.** Comisión de 0.125% por pata y operación, y préstamo de 0.25% anual sobre los cortos. Partición cronológica 60/20/20, con umbrales elegidos solo en entrenamiento.

PENDIENTE: grupo elegido, resultados principales por partición y respuesta a "¿la relación es estructural o un artefacto?".

## Instalación
Probado con Python 3.12. PENDIENTE: confirmar en las 3 compus.

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS:     source .venv/bin/activate
pip install -r requirements.txt
```

Los precios están congelados en `data/raw/prices.csv`. `python main.py` no descarga nada.

## Reproducir los resultados
```bash
python main.py          # pipeline completo; escribe artifacts/
python main.py --quick  # corrida rápida que no sobrescribe artifacts/ ni docs/
pytest                  # las 4 pruebas obligatorias y las unitarias
```
PENDIENTE: duración de la corrida completa y lista de lo que escribe en `artifacts/`.

## Semilla aleatoria
`SEED = 42`, definida en `config.py`. PENDIENTE: lista de los lugares donde se usa.

## Dashboard
```bash
streamlit run dashboard/app.py
```
Lee los archivos de `artifacts/` y no vuelve a correr el backtest. Si falta un archivo, el panel correspondiente lo dice y pide correr `python main.py`.

Para tener una copia estática que se abre sin Streamlit:
```bash
python -m dashboard.export   # escribe docs/dashboard.html
```

### Guía de uso
**Controles (fila de arriba).** Aplican a todas las gráficas y a la bitácora.
- **Vista:** cambia entre la vista ejecutiva y la técnica.
- **Particiones:** elige qué tramos mostrar: entrenamiento, prueba o validación.

En cada gráfica:
- Arrastra para hacer zoom; doble clic para regresar.
- Pasa el cursor para ver los valores.
- Da clic en la leyenda para ocultar o mostrar una serie.
- **Ver datos** abre la tabla con los valores de esa gráfica.

**Vista ejecutiva** (para el portfolio manager):
1. **Tarjeta de desempeño:** Sharpe, Sortino, Calmar, drawdown máximo, win rate, profit factor, retorno total y número de operaciones de la partición elegida. *Todas las particiones* las compara lado a lado.
2. **Curva de capital y drawdown:** el capital arriba y el drawdown abajo, con el mismo eje de tiempo. Las líneas verticales marcan dónde empiezan la prueba y la validación.
3. **Atribución de costos:** cómo el P&L bruto se convierte en P&L neto después de comisiones y préstamo.
4. **Retornos por operación:** histograma con la normal ajustada encima.
5. **Bitácora de operaciones:** se filtra por dirección (spread largo o corto) y por resultado (ganadora o perdedora).

**Vista técnica** (para el investigador cuantitativo o el gestor de riesgos):
1. **Spread S_t:** incluye la media que estima el Filtro 2. Las entradas largas (▲), las cortas (▼) y las salidas (✕) se distinguen por forma y por color.
2. **z-score con umbrales:** muestra los niveles ±z_in, ±z_out y ±z_stop.
3. **Pesos β̂_t del Filtro 1:** una línea por activo. La pata normalizada vale 1 y no se grafica.
4. **Primer eigenvector de Johansen móvil:** con la misma normalización que los pesos.
5. **Ganancia de Kalman de cada filtro:** indica qué tan rápido aprende el filtro.
6. **Varianza posterior de cada filtro:** en escala logarítmica; muestra la convergencia.
7. **Sharpe móvil de 252 días:** cubre toda la muestra.

Cada activo conserva su color en todas las gráficas. La paleta pasó las pruebas de daltonismo y contraste en modo claro y oscuro.

## Estructura
```
main.py            pipeline completo (python main.py)
config.py          parámetros del laboratorio y semilla
data/              ingesta, limpieza, validación y precios congelados
cointegration/     screening, Engle-Granger, Johansen y spread
kalman/            Filtro 1 (pesos) y Filtro 2 (señal)
strategy/          VECM, z-score, señales y tamaño de posición
backtest/          motor con costos y walk-forward
performance/       métricas
dashboard/app.py   dashboard en Streamlit
artifacts/         salidas del backtest que lee el dashboard
project_management/ charter, sprints y postmortem
tests/             pruebas con pytest
docs/              reporte.pdf, Powell y guía del dashboard
```

## Uso de asistencia de IA
PENDIENTE: cada integrante describe aquí con exactitud dónde usó IA.

- **JP:** Claude (Anthropic) para revisar el enunciado y las diapositivas, comparar los planes del equipo, redactar el charter, los sprints y este README, y escribir un primer borrador de `strategy/` y sus pruebas. JP las revisó, las ejecutó y responde por cada línea.
- **Gian:** PENDIENTE.
- **Ian:** PENDIENTE.

Cada integrante es responsable de todo el código entregado.
