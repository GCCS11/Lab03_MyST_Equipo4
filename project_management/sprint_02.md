# Sprint 02: del 2-nov al 8-nov de 2026

> Este archivo es el **tablero del equipo**. Cada integrante actualiza solo sus renglones: pendiente, en curso, listo (con su PR) o bloqueado (y por qué). El objetivo se escribió el 4-nov, a mitad de la semana.

## Objetivo
Entregar el pipeline completo, reproducible con `python main.py`:
- un grupo de 6 utilities seleccionado solo con entrenamiento;
- los dos filtros de Kalman, el VECM y la regla de señales;
- un backtest con costos;
- el dashboard, las 4 pruebas y el reporte.

Cada integrante debe terminar la semana con commits propios en su parte, y el equipo debe poder sostener con cifras propias si la relación encontrada es estructural o un artefacto de haber probado muchos grupos.

## Decisiones de la semana
| Fecha | Decisión | Quién |
|---|---|---|
| 4-nov | Grupo de **6 activos** (2 por integrante) | Equipo |
| 4-nov | **Sin Risk Parity** (fuera el 3.7, la figura 9 y la parte B de la prueba 4) | Equipo |
| 4-nov | Propuestas para avanzar. Se dan por buenas si nadie objeta antes del jue 5 a las 12:00 | JP |

Las propuestas del 4-nov:
- log-precios;
- ventana de 2014 a 2026;
- universo de 10 utilities;
- correlación sobre retornos;
- fechas fijas de partición;
- selección por estabilidad en subventanas;
- Filtro 1 cuasi-diferenciado;
- z previo;
- señal desde la desviación de equilibrio;
- regla con candado después del stop;
- ejecución al cierre de t+1.

## Tareas
### Gian: datos, selección, backtest y métricas
| ID | Tarea | Meta | Estado |
|---|---|---|---|
| G0 | Embudo v1: universos, ingesta, partición, I(1), Engle-Granger, Johansen, 120 grupos | 4-nov | Listo en la rama `gian` (commits `e1c7daa` a `1c6ec90`) |
| G1 | PR de `gian` a `main` | mié 4 | Pendiente |
| G2 | Embudo corregido: tamaño 6, 10 utilities, correlación sobre retornos, fechas fijas, 5 tickers nuevos en el snapshot | jue 5 | Pendiente |
| G3 | Etapa de estabilidad, regla entre grupos, tabla por columna de β y `artifacts/seleccion.json` | jue 5, 12:00 | Pendiente |
| G4 | Johansen móvil en `artifacts/eigvec_movil.csv` (figura 4) | vie 6 | Pendiente |
| G5 | Motor de backtest de n patas con comisión y préstamo; prueba 3 (contabilidad y golden-file) | vie 6, 12:00 | Pendiente |
| G6 | Walk-forward dentro de entrenamiento | vie 6 | Pendiente |
| G7 | Métricas y Sharpe móvil de 252 días | sáb 7 | Pendiente |
| G8 | Sección del reporte y figuras 1, 4, 7 y 8 | sáb 7 | Pendiente |

### Ian: filtros de Kalman y Powell
| ID | Tarea | Meta | Estado |
|---|---|---|---|
| I0 | Filtros 1 y 2 v1 con pruebas sintéticas; comparación de 5 sectores | 4-nov | En curso, en una rama local sin subir |
| I1 | Subir `feature/ian-kalman` en commits por tema y abrir el PR | mié 4 | Pendiente |
| I2 | Ajustar los filtros: log-precios, fechas fijas, grupo de `seleccion.json`, inicialización y burn-in | jue 5 | Pendiente |
| I3 | Elegir Q, R, φ y λ en entrenamiento; `artifacts/filtros.json` | vie 6 | Pendiente |
| I4 | Salidas con las columnas acordadas: pesos, ganancias, varianzas, innovaciones y z | jue 5 | Pendiente |
| I5 | Prueba 2 (normalización) y prueba 4 (filtro) | vie 6 | Pendiente |
| I6 | Powell en `docs/powell.md`: Filtro 1, Filtro 2 y sistema completo, con ejemplos numéricos | sáb 7 | Pendiente |
| I7 | Figuras 3 y 5 | sáb 7 | Pendiente |
| I8 | Armar `docs/reporte.pdf` | dom 8, 18:00 | Pendiente |

### JP: estrategia, pipeline, dashboard y gestión
| ID | Tarea | Meta | Estado |
|---|---|---|---|
| J1 | Charter, sprint 01 y sprint 02 | mié 4 | Listo (PR #4, rama `feature/jp`) |
| J2 | Esqueleto del README | jue 5 | Listo (PR #4) |
| J3 | `strategy/vecm.py`: VECM, βᵀα, vida media y desviación de equilibrio, con su prueba | jue 5 | Listo (PR #4): 6 pruebas en verde, incluida la comparación contra statsmodels |
| J4 | `strategy/signals.py`: regla con candado, stop y h_max, con su prueba | jue 5 | Listo (PR #4): 10 pruebas en verde, entre ellas candado, h_max, nunca dos posiciones y causalidad por truncamiento |
| J5 | `strategy/sizing.py`: 80% proporcional a abs(β), rebalanceo diario contra banda | vie 6 | Listo (PR #4): 8 pruebas en verde. Falta medir el costo de cada política con el motor de Gian (J6) |
| J6 | Búsqueda de umbrales en entrenamiento; `artifacts/theta.json` congelado con commit | vie 6 | Pendiente |
| J7 | `main.py` con `run_pipeline`, que escribe `artifacts/` | vie 6 | Pendiente |
| J8 | Una sola corrida de prueba y validación con θ congelado | vie 6, noche | Pendiente |
| J9 | Prueba 1 (causalidad) | sáb 7 | Pendiente |
| J10 | `dashboard/app.py` con vista ejecutiva y técnica; exportación y guía en `docs/` | sáb 7 | En curso (PR #4): las dos vistas, la exportación y la guía del README funcionan con artifacts sintéticos (7 pruebas). Falta correrlo con los artifacts reales de `main.py` (J7) y generar `docs/dashboard.html` |
| J11 | Sección del reporte y figuras 2 y 6 | sáb 7 | Pendiente |
| J12 | README final con el uso de IA de los tres, y el postmortem | dom 8 | Pendiente |

### Todos
| ID | Tarea | Meta | Estado |
|---|---|---|---|
| T1 | Preguntas al profe: prueba 4, "residual del VECM", fecha de exposición, cuarto entregable, grupo de Canvas | jue 5 | Pendiente |
| T2 | Clon limpio en las 3 compus: `pip install`, `python main.py`, `pytest` y dashboard | sáb 7 | Pendiente |
| T3 | Congelar a las 20:00, entregar en Canvas (Ian también) y ensayar la exposición de 10 minutos | dom 8 | Pendiente |

## Bloqueos y desviaciones
**4-nov:**
- Las decisiones de grupo de 6 y sin Risk Parity cambian el universo: ninguno de los 5 finalistas del embudo v1 tiene 6 activos.
- El filtro de correlación v1 usa log-precios; hay que pasarlo a retornos (S14 p16) antes de elegir el grupo.
- Ian y JP todavía no tienen commits en GitHub.
- En la exploración, el equipo ya vio resultados de prueba y validación. No se usarán para elegir, y se declara en el reporte.
