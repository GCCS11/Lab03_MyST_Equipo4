# Sección del reporte: filtros de Kalman

Lab 03, Equipo 4. Responsable: Ian Carlo Escalante Martínez. Las tres formulaciones de Powell están en `docs/powell.md`. Las cifras son solo de entrenamiento (2014-01-02 a 2021-09-08). Los resultados de prueba y validación se agregan después de la corrida única.

## Diseño

| Decisión | Filtro 1 | Filtro 2 |
|---|---|---|
| Estado | 5 pesos; AEP fija en +1 | Media latente y escala de la innovación |
| Entrada | Log-precios | Spread del Filtro 1, con pesos previos |
| Inicialización | β de Johansen; P0 de una pasada previa | Media de largo plazo del spread; P0 de una pasada previa |
| Q/R | 0.0001 | 0.0001 |
| R | 6.42×10⁻⁵ | 6.53×10⁻⁵ |
| φ | 0.925 | 0.904 |
| λ | No aplica | 0.99 |
| Burn-in | 60 días | 60 días; z vacío hasta que termina |

## Por qué cuasi-diferenciamos

La lámina S14 p18 trata el error de la observación como ruido blanco. Pero ese error es la desviación de equilibrio que queremos operar, y es persistente por construcción. S13 p17 indica que si las innovaciones salen autocorrelacionadas hay que subir Q. En nuestro caso, eso hace que los pesos persigan el spread y se pierda la señal. Por eso modelamos el error como AR(1).

| Filtro 1, Q/R = 0.0001 | Varianza innovaciones | Autocorrelación | Ljung-Box p | Vida media | Rango máximo de un peso |
|---|---|---|---|---|---|
| S14 p18 (φ = 0, c_t en el estado) | 0.416 | 0.823 | 0.000 | 3.8 días | 0.0266 |
| Cuasi-diferenciado (φ = 0.925) | 1.010 | −0.094 | 0.000 | 6.9 días | 0.0109 |

La misma lógica aplica al Filtro 2, y ahí se compara contra el nivel local en `docs/powell.md`. Esta decisión se aparta de la lámina y del plan inicial del equipo, y la sostenemos con estas cifras.

## Cómo elegimos los parámetros

El criterio se fijó antes de correr y está en `kalman/tuning.py`:

1. Restricción: vida media mediana de 5 a 60 días en las ventanas de evaluación.
2. Objetivo: mayor Sharpe neto concatenando esas ventanas, promediado entre dos juegos de umbrales provisionales.
3. Ventanas: ajuste desde 4 años y 4 ventanas de evaluación de 231 días. Johansen, φ, c y R se reestiman en cada una solo con datos previos.

Probamos 60 configuraciones: 5 de Q para el Filtro 1 y, para el Filtro 2, φ igual a 0 o estimada, 2 de λ y 3 de Q. Están en `artifacts/filtros_malla.csv`.

| Resultado del walk-forward | Valor |
|---|---|
| Mejor Sharpe neto | 0.66 |
| Vueltas completas de la mejor | 15 |
| Mediana de la malla | 0.03 |
| Configuraciones con Sharpe positivo | 33 de 60 |
| Con Sharpe positivo en los dos juegos de umbrales | 19 de 60 |

Con unos 3.7 años de evaluación, el error estándar de un Sharpe ronda 0.5, y además tomamos el mejor de 60. Este número sirve para ordenar configuraciones, no como evidencia de rentabilidad.

## Diagnóstico con los parámetros congelados

| Medida | Filtro 1 | Filtro 2 |
|---|---|---|
| Varianza de innovaciones normalizadas | 1.010 | 1.064 |
| Autocorrelación de primer orden | −0.094 | −0.087 |
| Ljung-Box a 10 rezagos, p | 0.000 | 0.000 |

La varianza está cerca de 1 y la autocorrelación es baja, pero el Ljung-Box rechaza ruido blanco en ambos filtros. La prueba 4 pasa con datos simulados del propio modelo y no pasa con datos reales. Lo declaramos como limitación: el AR(1) con varianza casi constante no captura toda la dinámica del spread.

## Figuras

Figura 3, en `docs/figures/fig3_pesos.png`. Los pesos casi no se mueven: el mayor rango es 0.0109, de XEL.

Figura 5, en `docs/figures/fig5_ganancias.png`. La ganancia del Filtro 1 baja durante 2014, se estabiliza y salta en marzo de 2020, cuando los precios caen de golpe y cambia `H_t`. La del Filtro 2 sube muy despacio de 0.0096 a 0.0100.

## Nuestra lectura

Lo que nos queda claro es que hay un intercambio directo: con una Q que conserva el spread, los pesos son casi fijos; con una Q que los mueve de verdad, el spread se deshace (a Q/R = 0.01, la vida media es de 0.7 días). Por eso no esperamos que el Filtro 1 rescate la relación si se rompe fuera de muestra. Su valor aquí es dar una estimación causal y medir la incertidumbre, no sostener el equilibrio. Si la estrategia pierde en validación, la defensa real son el stop y la tenencia máxima, no los pesos dinámicos.
