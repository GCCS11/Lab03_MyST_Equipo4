# Sección del reporte: datos, selección del grupo, backtest y métricas

Lab 03, Equipo 4. Área de Gian Carlo Campos Sayavedra (`data/`, `cointegration/`, `backtest/`, `performance/`); la selección final de 6 utilities es de Ian (`cointegration/seleccion.py`). Todas las cifras salen de `artifacts/`.

## 1. Datos y partición

- **Fuente:** precios de cierre ajustados por dividendos y splits de Yahoo Finance (`yfinance`, `auto_adjust=True`), diarios.
- **Snapshot:** se descargaron una vez y quedaron congelados en `data/raw/prices_utilities.csv`. Cubren 10 utilities, de 2014-01-02 a 2026-10-28 (3,225 días). `python main.py` no toca la red.
- **Escala:** se trabaja en log-precios. Los pesos son elasticidades y el número de acciones se calcula con el precio real (S12 p14, S14 p20).
- **Partición:** cronológica 60/20/20, con los mismos cortes en todo el pipeline (`data/loader.py`).

| Partición | Desde | Hasta |
|---|---|---|
| Entrenamiento | 2014-01-02 | 2021-09-08 |
| Prueba | 2021-09-09 | 2024-04-03 |
| Validación | 2024-04-04 | 2026-10-28 |

Las 10 series de log-precios son I(1) en entrenamiento:
- el ADF no rechaza raíz unitaria en niveles (p entre 0.24 y 0.97);
- el ADF sí rechaza en retornos (p < 0.001);
- el KPSS rechaza estacionariedad en niveles (p = 0.01).

## 2. Universo y embudo de selección

**Universo.** Primero exploramos 6 universos con justificación económica (`data/universe.py`): bancos, petroleras, utilities, bebidas, pagos y metales. Utilities reguladas fue el único sector con muchos finalistas. Su lógica económica es que tienen tarifas reguladas y la misma sensibilidad a tasas. Con eso fijamos un universo de 10 utilities (DUK, SO, AEP, D, XEL, ED, EXC, NEE, WEC y ES) y evaluamos sus 210 grupos de 6.

**Embudo** (S14 p16), solo con entrenamiento (`python -m cointegration.seleccion`). La regla se fijó antes de correr y está en el encabezado del módulo:

| Etapa | Criterio | Grupos que pasan |
|---|---|---|
| Combinaciones | 6 de 10 utilities | 210 |
| Correlación de retornos | Media entre pares de 0.75 a 0.84 en todos los grupos | 210 |
| Johansen por traza al 95% | Rango ≥ 1 | 146 |
| Spread admisible | Vector con ADF p < 0.05 y vida media de 5 a 60 días | 146 |
| Engle-Granger multivariado | p < 0.05; prioriza, no elimina | 94 |
| Estabilidad | Fracción de subventanas de 2, 3 y 4 años de entrenamiento que repiten Johansen y spread | Gana la mayor entre los que pasan Engle-Granger |

| Grupo (los 5 más estables) | Estabilidad | Vida media | Engle-Granger p |
|---|---|---|---|
| DUK, AEP, XEL, ED, NEE, ES | 0.70 | 12.3 días | 0.13 |
| DUK, AEP, XEL, ED, WEC, ES | 0.66 | 12.1 días | 0.06 |
| **AEP, XEL, ED, EXC, WEC, ES** | **0.65** | **8.9 días** | **0.015** |
| SO, AEP, XEL, ED, EXC, ES | 0.62 | 8.7 días | 0.013 |
| AEP, D, XEL, ED, WEC, ES | 0.62 | 21.7 días | 0.020 |

**El grupo elegido** es AEP, XEL, ED, EXC, WEC y ES (`artifacts/seleccion.json`):
- es el más estable entre los que pasan Engle-Granger;
- su vector de Johansen, normalizado en AEP, es (1, 0.411, −0.412, −0.187, −0.338, −0.598);
- su vida media es de 8.9 días.

**Registro.**
- Los 210 grupos están en `artifacts/seleccion_grupos.csv`, con todas sus etapas.
- Los 120 grupos del primer embudo de Gian están en `artifacts/screening_groups.csv`.
- En total se evaluaron 1,576 grupos, contando la exploración inicial de sectores. Ese número es el que importa para las pruebas múltiples (ver la sección de estrategia).

## 3. Johansen móvil (figura 4)

`python -m cointegration.eigvec_movil --hasta todo` estima Johansen en ventanas de 252 días con paso de 5 (595 ventanas). Cada ventana usa solo datos hasta su fecha final. El primer eigenvector se normaliza por norma L1 y se alinea en signo con la ventana anterior.

- **Figura 1** (`docs/figures/fig1_precios.png`): los precios de las 6 patas normalizados a 100, con los cortes de partición.
- **Figura 4** (`docs/figures/fig4_eigenvector.png`): los pesos del eigenvector cambian de forma marcada incluso dentro de entrenamiento.
- **Prueba de traza en ventanas de un año:** rebasa el valor crítico al 95% en 34% de las ventanas que terminan en entrenamiento, 6% en prueba y 17% en validación. Con un año de datos la prueba tiene poco poder, pero el patrón coincide con el diagnóstico por partición: la relación es intermitente.

## 4. Motor de backtest y métricas

**Motor** (`backtest/engine.py`). Es diario y de n patas:
- toma la dirección `d_t` de la regla, la ejecuta en t+1 y dimensiona con `strategy/sizing.py`;
- cobra 0.125% de comisión sobre el monto operado en cada pata, incluidos los rebalanceos;
- cobra 0.25% anual de préstamo sobre el valor de los cortos, día por día;
- cierra cualquier posición abierta al final de la serie.

La prueba 3 (`tests/test_contabilidad.py`) verifica cuatro cosas:
- que capital = efectivo + valor de mercado en cada paso;
- que el motor no mira precios futuros;
- que la banda rebalancea menos que el modo diario;
- que la salida de un escenario fijo coincide con un golden-file.

**Walk-forward** (`backtest/walk_forward.py`). Es la búsqueda de umbrales con 5 pliegues dentro de entrenamiento descrita en la sección de estrategia.

**Métricas** (`performance/metrics.py`):
- Sharpe y Sortino anualizados con √252;
- Calmar, drawdown máximo con sus fechas, win rate y profit factor;
- atribución de costos y Sharpe móvil de 252 días.

**Figuras:**
- **Figura 7** (`docs/figures/fig7_equity.png`): capital y drawdown continuos. El capital sube de forma escalonada hasta finales de 2023. En validación cae, sobre todo en dos escalones (finales de 2025 y abril de 2026) que corresponden a stops en corto.
- **Figura 8** (`docs/figures/fig8_retornos.png`): retornos por operación. La media es 0.09% y la desviación 0.77%, con asimetría de −0.69 en 40 operaciones. Las pérdidas por stop forman la cola izquierda.
