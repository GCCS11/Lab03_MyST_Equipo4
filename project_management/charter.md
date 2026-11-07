# Project charter: Lab 03, arbitraje estadístico con cointegración y filtros de Kalman

**Equipo 4:**
- Juan Pablo Barba González (JPBAG0)
- Gian Carlo Campos Sayavedra (GCCS11)
- Ian Carlo Escalante Martínez (ianmartinez01)

**Entrega del repositorio:** domingo 8-nov-2026 a las 23:59. Congelamos el código a las 20:00.

## Alcance

**La estrategia hará esto:**
- Seleccionar **un grupo de 6 activos** cointegrados (2 × n, con n = 3 integrantes). El grupo sale de un universo de 10 utilities eléctricas reguladas de EE. UU., con el embudo del 3.2. Todo se ejecuta solo sobre entrenamiento y se registran todos los grupos evaluados.
- Estimar los pesos dinámicos β̂_t con el Filtro 1, y la media y la varianza del spread con el Filtro 2. El z-score se calcula con los estadísticos previos a la actualización.
- Estimar el VECM con el vector de Johansen y generar señales desde la desviación de equilibrio.
- Operar con 80% del capital proporcional a abs(β̂_t), con comisión de 0.125% por pata y préstamo de 0.25% anual sobre los cortos.
- Comparar el rebalanceo diario contra una banda de no-rebalanceo.
- Evaluar en entrenamiento, prueba y validación (60/20/20 cronológico) con métricas, un dashboard en Streamlit y las 4 pruebas de `tests/`.

**No hará esto:**
- **Risk Parity entre varios grupos.** Queda fuera el 3.7, junto con la figura 9 y la parte B de la prueba 4.
- Operación intradía o en vivo.
- Modelar slippage, margen o restricciones de préstamo más allá de la tasa fija.
- Corregir el sesgo de supervivencia del universo: son empresas que hoy cotizan, y se declara como limitación.
- Elegir grupo, ventana, escala o parámetros mirando prueba o validación.

## Plan de hitos
| Hito | Responsable | Fecha objetivo |
|---|---|---|
| Estructura del repo, `config.py` y dependencias fijadas | Gian | 29-oct ✅ |
| Embudo de cointegración v1, con 120 grupos registrados | Gian | 4-nov ✅ |
| **Pipeline de datos completo:** snapshot con las 10 utilities y fechas fijas de partición | Gian | jue 5-nov |
| **Prueba de cointegración completa:** grupo final en `artifacts/seleccion.json` | Gian | jue 5-nov, 12:00 |
| **Filtro 1 validado** (pruebas 2 y 4) y Filtro 2 | Ian | vie 6-nov |
| VECM, regla de señales y tamaño de posición | JP | vie 6-nov |
| **Backtest completo:** motor con costos (prueba 3), walk-forward, θ congelado y una sola corrida de prueba y validación | Gian y JP | vie 6-nov, noche |
| **Dashboard funcionando** y prueba 1 (causalidad) | JP | sáb 7-nov |
| Reporte en `docs/reporte.pdf` | Ian (arma), todos (escriben) | dom 8-nov, 18:00 |
| Código congelado y entrega en Canvas | Todos | dom 8-nov, 20:00 |
| Exposición | Todos | por confirmar (probablemente lun 9-nov) |

## Registro de riesgos
| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| 1 | **Pérdida de cointegración fuera de muestra.** En la exploración ya pasó: ningún ganador fue estacionario en validación con el vector fijo | Alta | Alto | Pesos dinámicos (Filtro 1), stop sobre abs(z), tenencia máxima según la vida media y Johansen móvil. Si ocurre, se reporta como respuesta a "¿estructural o artefacto?" |
| 2 | **Falsos positivos por pruebas múltiples.** Al menos 1,366 grupos evaluados al 4-nov | Alta | Alto | Justificación económica previa. Universo y regla de selección commiteados antes de correr. Etapa de estabilidad en subventanas. Se reporta el total |
| 3 | **Divergencia o mala sintonía de los filtros.** Una Q alta hace que los pesos persigan el spread | Media | Alto | Q y R elegidos solo en entrenamiento, con la vida media como restricción. Diagnóstico de innovaciones normalizadas (prueba 4) |
| 4 | **Look-ahead:** z posterior, partición por número de renglones, elegir con prueba | Media | Alto | z con prior, fechas fijas en `config.py`, θ congelado y commiteado antes de evaluar, y prueba 1 de causalidad |
| 5 | **Calidad de datos:** ajustes, spin-off de EXC en 2022, huecos | Baja | Medio | Snapshot congelado en `data/raw/`, revisión de huecos y de saltos mayores a 15% (al 4-nov solo aparece el COVID de marzo de 2020) |
| 6 | **Integrante sin commits o integración tardía** (0 en la parte técnica para ese integrante) | Media | Alto | Una rama por integrante, PR diarios con merge commit, contratos de interfaz y revisión de `git shortlog` el sábado |
| 7 | **El repo no corre en otra compu** (−10 puntos) | Media | Medio | `requirements.txt` fijado (se resuelve en Python 3.12), clon limpio en las 3 compus el sábado y `python main.py` sin internet |
| 8 | **Entrega en Canvas ligada a un grupo viejo:** "Lab 03 - Entrega" usa "Equipos Labs" (JP y Gian) | Media | Alto para Ian | Preguntarle al profe. Ian entrega también la liga |

## Roles
Cada quien hace commits **con su cuenta y en su rama** (`feature/<nombre>-<tema>`), con PR a `main` que revisa otro integrante.

| Integrante | Responsabilidad | Carpetas y archivos (contribución esperada en Git) | Pruebas | Reporte y exposición |
|---|---|---|---|---|
| **Gian** | Datos, selección del grupo, backtest y métricas | `config.py`, `data/`, `cointegration/`, `backtest/`, `performance/` | 3 (contabilidad de n patas y golden-file) | Universo y embudo, "¿estructural o artefacto?", resultados y costos. Figuras 1, 4, 7 y 8 |
| **Ian** | Filtros de Kalman, formulación de Powell y armado del reporte | `kalman/`, `docs/powell.md`, `docs/reporte.pdf` | 2 (normalización) y 4 (filtro) | Los dos filtros y las 3 formulaciones de Powell. Figuras 3 y 5 |
| **JP** | Estrategia, pipeline, dashboard y gestión | `strategy/`, `main.py`, `dashboard/`, `README.md`, `project_management/` | 1 (causalidad) | VECM, regla, tamaño de posición, rebalanceo y dashboard. Figuras 2 y 6 |

Cualquier integrante debe poder explicar cualquier parte. El sábado cada quien expone una parte de otro como ensayo.
