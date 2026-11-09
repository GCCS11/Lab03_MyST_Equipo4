# Sección del reporte: estrategia, resultados y "¿estructural o artefacto?"

Lab 03, Equipo 4. Responsable: Juan Pablo Barba González. Código: `strategy/` (VECM, señales, tamaño de posición y theta), `main.py` y `cointegration/diagnostico.py`. Todas las cifras salen de `artifacts/`.

## 1. VECM (sección 3.6)

El VECM se estima solo en entrenamiento, con el vector de Johansen del grupo como término de corrección de error y un rezago en diferencias (`strategy/vecm.py`, `artifacts/vecm.json`).

| Medida | Valor |
|---|---|
| βᵀα | −0.066 |
| Condición −2 < βᵀα < 0 | Se cumple: hay corrección de error |
| Vida media del modelo, ln(0.5)/ln(1 + βᵀα) | 10.1 días |
| Vida media del spread por regresión AR(1) (selección) | 8.9 días |
| α por activo | AEP 0.010, XEL 0.042, ED 0.056, EXC 0.037, WEC 0.066, ES 0.068 |

- **Quién corrige.** AEP casi no se ajusta (α = 0.010, la menor). Cuando el spread se aleja, son las otras cinco patas las que regresan al equilibrio. Esto anticipa lo que pasó fuera de muestra: si AEP se mueve por su cuenta, nadie la jala de regreso.
- **Qué alimenta la señal.** Tomamos el "residual del VECM" como la desviación de equilibrio `S_t = β̂_tᵀ log P_t − c`, con los pesos **previos** del Filtro 1. Esa serie entra al Filtro 2, que da el z-score causal. El término de corrección de error es lo que revierte; el error de pronóstico ε_t del VECM es casi ruido y no tiene nada que operar.
- **Para qué sirve la vida media.** Calibra la tenencia máxima: `h_max` se buscó como 1, 2 y 3 veces la vida media de selección (9, 18 y 27 días).

## 2. Regla de señales y tamaño de posición

**Regla** (`strategy/signals.py`, basada en S12 p20):

| Estado | Condición sobre z_t |
|---|---|
| Entrar largo en el spread | z_t < −z_in y abs(z_t) < z_stop |
| Entrar corto en el spread | z_t > z_in y abs(z_t) < z_stop |
| Salir por reversión | abs(z_t) < z_out |
| Salir por stop | abs(z_t) > z_stop |
| Salir por tiempo | La posición cumple h_max días |

- **Candado después del stop.** Tal como viene en la lámina, la regla reabriría al día siguiente de un stop si z sigue arriba de z_in. Nosotros exigimos que abs(z) baje de z_out antes de volver a entrar.
- **Ejecución.** La decisión se toma con el z del cierre de t y se ejecuta al precio de t+1. La prueba 1 (causalidad) verifica que truncar los datos en cualquier fecha no cambia ninguna señal anterior.

**Tamaño de posición** (`strategy/sizing.py`): se invierte 80% del capital. Cada pata recibe `N_i = 0.8 · V · abs(β_i) / Σ abs(β_j)` dólares, con el signo de su peso y de la dirección. Para no pagar comisión todos los días por movimientos mínimos de β̂_t, una pata se rebalancea solo cuando se aleja de su objetivo más que una banda, medida como fracción del capital.

**Costos** (motor de Gian, `backtest/engine.py`): 0.125% de comisión sobre el monto operado en cada pata, incluidos los rebalanceos, y 0.25% anual de préstamo sobre los cortos.

## 3. Cómo elegimos los umbrales

- **Malla:** 162 combinaciones.
  - z_in en {1.0, 1.5, 2.0}, z_out en {0.25, 0.5} y z_stop en {2.5, 3.0, 3.5};
  - h_max en {9, 18, 27} días;
  - banda en {0, 0.75%, 1.5%}.
- **Evaluación:** walk-forward con 5 pliegues, todos dentro de entrenamiento (`backtest/walk_forward.py`, de Gian).
- **Regla, fijada antes de ver resultados** (`strategy/theta.py`):
  - se ordena por el Sharpe medio entre pliegues menos una desviación estándar;
  - solo cuentan los theta con al menos 3 operaciones en cada pliegue.
- **Ganador:** z_in 2.0, z_out 0.5, z_stop 3.0, h_max 27 y banda de 0.75%.
  - Sharpe por pliegue: 0.34, 0.91, 0.07, 0.37 y 0.85.
  - Media 0.50 y desviación 0.32.
  - El de mayor media (0.55) tenía más dispersión.
- **El resto de la malla:** 110 de las 162 combinaciones tienen Sharpe medio positivo, y la mediana es 0.14. Nuestro número se acerca al mejor de 162 intentos, así que lo esperamos inflado.
- **Congelado:**
  - `artifacts/theta.json` se commiteó (`4557c17`, tag `theta-congelado`) **antes** de correr prueba y validación;
  - `python -m strategy.theta` se niega a sobrescribirlo;
  - prueba y validación se corrieron una sola vez.

## 4. Resultados de la corrida única

El backtest es continuo desde el primer z válido: cada partición arranca con el capital que heredó de la anterior, y cada operación cuenta en la partición donde se abrió. Capital inicial de 1,000,000 USD.

| | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| Fechas | 2014-01 a 2021-09 | 2021-09 a 2024-04 | 2024-04 a 2026-10 |
| Sharpe | 0.59 | 0.32 | −0.88 |
| Sortino | 0.96 | 0.47 | −0.98 |
| Calmar | 0.37 | 0.40 | −0.35 |
| Drawdown máximo | 2.2% | 1.0% | 4.0% |
| Retorno total | 6.4% | 1.0% | −3.6% |
| Retorno anualizado | 0.8% | 0.4% | −1.4% |
| Volatilidad anualizada | 1.4% | 1.2% | 1.6% |
| Operaciones | 28 | 6 | 6 |
| Win rate | 79% | 67% | 33% |
| Profit factor | 2.36 | 1.96 | 0.25 |
| P&L bruto (USD) | 123,838 | 23,980 | −25,493 |
| Comisiones (USD) | 58,466 | 13,097 | 12,773 |
| Préstamo (USD) | 1,471 | 342 | 288 |
| P&L neto (USD) | 63,901 | 10,541 | −38,554 |

**Cómo cerró cada operación** (P&L neto en USD):

| Motivo | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| Reversión | 22 (+110,920) | 4 (+21,568) | 1 (+8,165) |
| Stop | 4 (−35,399) | 1 (−7,227) | **5 (−46,719), todas en corto** |
| h_max | 2 (−11,620) | 1 (−3,799) | 0 |

- **Los costos se comen la mitad.** Las comisiones son 47% del P&L bruto en entrenamiento y 55% en prueba. El préstamo pesa poco (1%). La ventaja bruta existe, pero es delgada: con una comisión el doble de alta, entrenamiento quedaría cerca de cero.
- **Escala.** La volatilidad es baja (1.2% a 1.6% anual) porque el spread de utilities se mueve poco. Un Sharpe de 0.59 equivale a menos de 1% anual.
- **Validación.** El z se queda arriba de +2 casi todo el tramo. La regla abre cortos una y otra vez y el stop los cierra: cinco stops seguidos. El candado evita que reabra al día siguiente, pero no evita que reabra cuando z regresa por un momento a la zona de entrada. El stop y el tamaño de 80% limitaron la pérdida a −3.6%, con un drawdown de 4%.

**Figuras de esta sección:**
- **Figura 2** (`docs/figures/fig2_spread.png`): el spread con la media del Filtro 2 y las 18 entradas largas, las 22 cortas y las 40 salidas.
- **Figura 6** (`docs/figures/fig6_zscore.png`): el z-score con ±z_in, ±z_out y ±z_stop.
- **Figuras 7 y 8**, de Gian: el capital y la distribución de retornos por operación.

En las figuras 2 y 6 se ve lo mismo que en la tabla. Hasta 2021 el spread oscila alrededor de −0.40. Después sube a cerca de −0.30, y la media del Filtro 2 lo sigue muy despacio.

## 5. ¿La relación es estructural o un artefacto?

**Diagnóstico por partición**, con el vector de entrenamiento fijo (`python -m cointegration.diagnostico`, `artifacts/diagnostico_particiones.csv`). Corre solo con `theta.json` congelado y no decide nada del pipeline.

| | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| ADF del spread (p) | 0.000 | 0.27 | 0.36 |
| Vida media | 8.9 días | 46 días | 64 días |
| Johansen re-estimado en el tramo (rango) | 1 | 0 | 0 |
| Ventanas móviles de 2 años con relación | 58% | 20% | 3% |
| Volatilidad del spread | 0.021 | 0.043 | 0.076 |

**Prueba anidada** (`artifacts/prueba_anidada.json`). Partimos entrenamiento otra vez en 60/20/20 y repetimos la selección completa adentro, usando solo entrenamiento.

| | Test interno (2018-08 a 2020-02) | Validación interna (2020-02 a 2021-09) |
|---|---|---|
| Candidatos internos que siguen estacionarios con su vector | 3 de 75 (4%) | 39 de 75 (52%) |
| Ganador interno (DUK, AEP, XEL, ED, NEE, ES): ADF p y vida media | 0.12, 28 días | 0.06, 8 días |
| Nuestro grupo con el vector interno: ADF p y vida media | 0.26, 41 días | 0.001, 5 días |

**Pruebas múltiples.** Evaluamos 1,576 grupos en total: la exploración de 6 universos y los 210 grupos de 6 de las 10 utilities. Si ninguno estuviera cointegrado, al 5% esperaríamos unos 79 falsos positivos. La prueba anidada confirma el problema: que un grupo pase el embudo dentro de muestra casi no predice que siga cointegrado en el tramo siguiente (3 de 75 es lo que daría el azar).

**Nuestra respuesta: ninguna de las dos en forma pura.**

1. **No es un artefacto puro.** El grupo tiene una razón económica (utilities reguladas con la misma sensibilidad a tasas). La relación aparece en Johansen re-estimado dentro de entrenamiento y en 58% de las ventanas de 2 años. También vuelve en la validación interna (p = 0.001), y el VECM corrige error con una vida media razonable.
2. **Tampoco es estructural.** Depende del periodo: aparece y desaparece (se rompe en el test interno y vuelve en la validación interna). Después de 2021 se deshace por completo: 3% de las ventanas, vida media de 64 días y volatilidad del spread 3.6 veces la de entrenamiento. El Johansen móvil (figura 4) muestra un eigenvector inestable incluso dentro de entrenamiento.
3. **El problema es de selección.** Con 1,576 grupos probados, el ganador dentro de muestra es en parte el que mejor se ajustó al ruido de ese periodo.

**Quién rompe la relación.** Entre la media de entrenamiento y la de validación, el spread sube 0.19. La pata de AEP aporta +0.72 y las demás lo compensan solo en parte. AEP es justo la pata que el VECM dice que no corrige (α más pequeña). Hipótesis que no comprobamos: el choque de tasas de 2022 afectó distinto a cada utility, y EXC separó Constellation en febrero de 2022.

**Qué haríamos para operarla.** No cambiar de grupo mirando prueba: eso sería sobreajuste. Volveríamos a probar la relación de forma causal (S14 p21), por ejemplo con Johansen móvil o ADF sobre una ventana que termina en t, y apagaríamos la estrategia cuando la relación se cae. En validación, eso habría evitado los cinco stops seguidos.

## 6. Limitaciones

- **Poca muestra.** 40 operaciones en total y 6 en cada tramo fuera de muestra. Con 2.5 años, el error estándar de un Sharpe ronda 0.6, así que la diferencia entre 0.32 y −0.88 no es significativa por sí sola. Lo que sostiene la conclusión es el diagnóstico de la sección 5, no el Sharpe.
- **Un solo grupo.** Sin Risk Parity, no hay diversificación entre grupos que amortigüe la ruptura de uno.
- **Universo con sesgo de supervivencia.** Las 10 utilities son las grandes de hoy; las que dejaron de cotizar no estaban en la lista.
- **Precios de cierre.** Se ejecuta al cierre de t+1 sin deslizamiento. Con la comisión de 0.125% el costo ya domina; un deslizamiento real lo empeoraría.
- **Pesos casi fijos.** El Filtro 1 mueve los pesos muy poco (figura 3), así que no rescata la relación cuando se rompe. Ver la sección de Ian.
