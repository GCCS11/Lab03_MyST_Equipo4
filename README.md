# Lab03_MyST_Equipo4: arbitraje estadístico con cointegración y filtros de Kalman

Microestructura y Sistemas de Trading, ITESO, otoño 2026. Laboratorio 03.

**Reporte:** `docs/reporte.pdf`. **Dashboard estático:** `docs/dashboard.html`.

## Integrantes
| Integrante | GitHub | Responsabilidad |
|---|---|---|
| Juan Pablo Barba González | JPBAG0 | Estrategia (VECM, señales, tamaño de posición), `main.py`, dashboard y gestión del proyecto |
| Gian Carlo Campos Sayavedra | GCCS11 | Datos, selección del grupo (cointegración), backtest y métricas |
| Ian Carlo Escalante Martínez | ianmartinez01 | Filtros de Kalman, formulación de Powell y reporte |

## Nivel de alcance
**Un grupo de 6 activos** (2 × n, con n = 3 integrantes), **sin Risk Parity**. No se implementan la sección 3.7, la figura 9 ni la parte de nivel B de la prueba 4. El enunciado no define los niveles; este es el alcance que declaramos.

## Descripción
Estrategia de arbitraje estadístico sobre un portafolio de 6 utilities eléctricas reguladas de EE. UU., con datos diarios de 2014 a 2026. El flujo es:
1. **Selección del grupo.** Se aplica el embudo del 3.2 solo sobre entrenamiento: universo con justificación económica, correlación móvil de retornos, Engle-Granger, Johansen, validación del spread y estabilidad. Se registran todos los grupos evaluados.
2. **Filtro 1 (Kalman).** Estima los pesos dinámicos β̂_t del portafolio reversivo, con la pata normalizada en +1.
3. **VECM.** Se estima con el vector de Johansen como término de corrección de error. Da la velocidad de ajuste βᵀα y la vida media que calibra la tenencia máxima.
4. **Filtro 2 (Kalman).** Estima la media y la varianza latentes de la desviación de equilibrio. El z-score usa los estadísticos previos a la actualización.
5. **Regla y tamaño de posición.** Se entra si z_in < abs(z) < z_stop y se sale si abs(z) < z_out, si se activa el stop o si se llega a la tenencia máxima. Se usa 80% del capital, repartido en proporción a abs(β̂_t).
6. **Backtest.** Comisión de 0.125% por pata y operación, y préstamo de 0.25% anual sobre los cortos. Partición cronológica 60/20/20, con umbrales elegidos solo en entrenamiento.

**Grupo elegido:** AEP, XEL, ED, EXC, WEC y ES, con AEP como pata normalizada. Se eligió solo con entrenamiento, entre 210 grupos de 6 de 10 utilities (1,576 grupos evaluados en total contando la exploración).

**Umbrales congelados** (`artifacts/theta.json`, elegidos con el walk-forward de entrenamiento antes de correr prueba y validación): z_in 2.0, z_out 0.5, z_stop 3.0, h_max 27 días y banda de rebalanceo de 0.75%.

**Resultados de la corrida única** (capital inicial de 1,000,000 USD, con costos):

| | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| Sharpe | 0.59 | 0.32 | −0.88 |
| Retorno total | 6.4% | 1.0% | −3.6% |
| Drawdown máximo | 2.2% | 1.0% | 4.0% |
| Operaciones | 28 | 6 | 6 |
| Comisiones (USD) | 58,466 | 13,097 | 12,773 |

En validación, 5 de las 6 operaciones cerraron por stop. Las comisiones se comen cerca de la mitad del P&L bruto en entrenamiento.

**¿La relación es estructural o un artefacto?** Ninguna de las dos en forma pura. Es una relación real pero que depende del periodo, y su selección está inflada por las pruebas múltiples. El diagnóstico usa el vector de entrenamiento fijo (`python -m cointegration.diagnostico`):

| | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| ADF del spread (p) | 0.000 | 0.27 | 0.36 |
| Vida media | 8.9 días | 46 días | 64 días |
| Ventanas de 2 años con relación de Johansen | 58% | 20% | 3% |
| Volatilidad del spread | 0.021 | 0.043 | 0.076 |

- **Prueba anidada.** Repetimos la selección dentro de entrenamiento (60/20/20 interno). Solo 3 de 75 candidatos siguen estacionarios en el test interno, lo mismo que daría el azar. Nuestro grupo se rompe ahí y vuelve en la validación interna.
- **Quién rompe la relación.** AEP explica la mayor parte del desplazamiento del spread después de 2021, y es la pata que el VECM dice que no corrige.
- **Qué haríamos.** No cambiar de grupo mirando prueba. Volveríamos a probar la relación de forma causal y apagaríamos la estrategia cuando se cae.

El detalle está en `docs/reporte.pdf`, sección de estrategia.

## Instalación
Probado con Python 3.12.10 en Windows 11, desde un clon limpio con un entorno nuevo (8-nov).

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS:     source .venv/bin/activate
pip install -r requirements.txt
```

Los precios están congelados en `data/raw/prices_utilities.csv` (las 10 utilities, de 2014-01-02 a 2026-10-28). `python main.py` no descarga nada.

## Reproducir los resultados
```bash
python main.py          # pipeline completo; escribe artifacts/
python main.py --quick  # corrida rápida que no sobrescribe artifacts/ ni docs/
pytest                  # las 4 pruebas obligatorias y las unitarias
```
`python main.py` tarda unos 3 segundos. Lee lo que ya está congelado (`seleccion.json`, `filtros.json` y `theta.json`) y escribe en `artifacts/`:
- `precios.csv`, `senales.csv`, `pesos.csv`, `ganancias.csv` y `varianzas.csv`: la vista técnica;
- `equity.csv`, `operaciones.csv` y `metricas.json`: el backtest y las métricas por partición;
- `vecm.json`: α, βᵀα y la vida media del VECM en entrenamiento.

Cada parámetro congelado tiene su propio comando. Ya se corrieron en este orden y sus salidas están en el repo:

| Paso | Comando | Escribe |
|---|---|---|
| 1. Selección del grupo (solo entrenamiento) | `python -m cointegration.seleccion` | `seleccion.json`, `seleccion_grupos.csv` |
| 2. Parámetros de los filtros (solo entrenamiento) | `python -m kalman.tuning` | `filtros.json`, `filtros_malla.csv` |
| 3. Walk-forward de umbrales (solo entrenamiento) | `python -m backtest.walk_forward` | `theta_busqueda.csv` |
| 4. Congelar theta (una sola vez) | `python -m strategy.theta` | `theta.json` |
| 5. Pipeline y backtest | `python main.py` | ver arriba |
| 6. Johansen móvil en toda la muestra | `python -m cointegration.eigvec_movil --hasta todo` | `eigvec_movil.csv` |
| 7. Diagnóstico por partición y prueba anidada | `python -m cointegration.diagnostico` | `diagnostico_particiones.csv`, `prueba_anidada.json` |
| 8. Figuras y dashboard estático | `python -m cointegration.figures`, `python -m kalman.figures --completo`, `python -m strategy.figures`, `python -m performance.figures`, `python -m dashboard.export` | `docs/` |

`python -m strategy.theta` se niega a sobrescribir `theta.json`, y los pasos 6 y 7 se niegan a correr sin él, porque leen prueba y validación.

## Semilla aleatoria
`SEED = 42`, definida en `config.py`. `main.py` la fija al arrancar (`set_seed`), y `kalman/simulation.py` la usa para los datos simulados de la prueba 4. El resto del pipeline es determinista: no tiene ningún paso aleatorio.

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
**Controles.** La vista se elige en la barra lateral; las particiones, arriba de las gráficas. Aplican a todas las gráficas y a la bitácora.
- **Vista (barra lateral):** cambia entre la presentación, la vista ejecutiva y la técnica. El tema es oscuro, con paneles de vidrio y entradas animadas que se apagan con movimiento reducido (`dashboard/style.py`).
- **Particiones:** elige qué tramos mostrar: entrenamiento, prueba o validación. No aparece en la presentación, que siempre muestra la muestra completa.

**Presentación.** Es el guion de la exposición en 10 pasos.
- Se avanza con "Anterior" y "Siguiente", o con los números de arriba.
- Cada paso tiene un título que dice la conclusión, quién expone, sus cifras clave y una gráfica.
- Las notas del expositor están cerradas, para no leerlas en pantalla.
- El código está en `dashboard/presentation.py`, y lee también `seleccion_grupos.csv`, `filtros.json`, `vecm.json`, `theta_busqueda.csv`, `diagnostico_particiones.csv` y `prueba_anidada.json`.

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
docs/              reporte.pdf (y sus secciones en .md), Powell, figuras y dashboard.html
```

## Uso de asistencia de IA
Cada integrante describe aquí con exactitud dónde usó IA.

- **JP:** Claude (Anthropic). Lo usó para:
  - revisar el enunciado y las diapositivas, y comparar los planes del equipo;
  - redactar el charter, los sprints, el postmortem y este README;
  - escribir un primer borrador de `strategy/` y sus pruebas;
  - conectar el motor y las métricas en `main.py`, la regla para congelar theta y las figuras 2 y 6;
  - escribir `cointegration/diagnostico.py` y redactar las secciones `docs/reporte_estrategia.md` y `docs/reporte_datos.md`, y armar el PDF.

  JP revisó y ejecutó cada parte, y responde por cada línea.
- **Gian:** PENDIENTE.
- **Ian:** Claude Code (Anthropic). Lo usó para:
  - leer el enunciado y las láminas S11 a S14, y comparar sectores en la exploración inicial;
  - escribir los filtros 1 y 2 (`kalman/`), sus pruebas (pruebas 2 y 4) y los datos sintéticos;
  - escribir la selección del grupo (`cointegration/seleccion.py`) y el walk-forward que congela `filtros.json` (`kalman/tuning.py`);
  - redactar `docs/powell.md` y `docs/reporte_filtros.md`, generar las figuras 3 y 5, revisar el PR de JP y armar el PDF del reporte.

  Ian revisó y ejecutó cada parte, y responde por cada línea.

Cada integrante es responsable de todo el código entregado.
