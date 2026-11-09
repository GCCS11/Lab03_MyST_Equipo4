# Sección del reporte: filtros de Kalman

Lab 03, Equipo 4. Responsable: Ian Carlo Escalante Martínez. Las tres formulaciones de Powell están en `docs/powell.md`. Los parámetros se eligieron solo con entrenamiento (2014-01-02 a 2021-09-08) y quedaron congelados en `artifacts/filtros.json` antes de la corrida única de prueba y validación.

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

## Diagnóstico en entrenamiento

| Medida | Filtro 1 | Filtro 2 |
|---|---|---|
| Varianza de innovaciones normalizadas | 1.010 | 1.064 |
| Autocorrelación de primer orden | −0.094 | −0.087 |
| Ljung-Box a 10 rezagos, p | 0.000 | 0.000 |

La varianza está cerca de 1 y la autocorrelación es baja, pero el Ljung-Box rechaza ruido blanco en ambos filtros. La prueba 4 pasa con datos simulados del propio modelo y no pasa con datos reales. Lo declaramos como limitación: el AR(1) con varianza casi constante no captura toda la dinámica del spread.

## Resultados en prueba y validación

Misma corrida continua de `python main.py`, con los parámetros congelados. En entrenamiento se descartan los 60 días del burn-in.

| Medida | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| Días | 1,875 | 645 | 645 |
| Filtro 1: varianza de innovaciones normalizadas | 1.010 | 1.073 | 1.548 |
| Filtro 1: autocorrelación | −0.094 | 0.139 | −0.024 |
| Filtro 1: Ljung-Box, p | 0.000 | 0.000 | 0.006 |
| Filtro 2: varianza de innovaciones normalizadas | 1.064 | 1.092 | 0.916 |
| Filtro 2: autocorrelación | −0.087 | 0.053 | −0.071 |
| Filtro 2: Ljung-Box, p | 0.000 | 0.000 | 0.000 |
| Vida media del spread | 6.9 días | 28.4 días | 14.2 días |
| ADF del spread, p | 0.000 | 0.111 | 0.017 |
| Media de z | 0.05 | 0.13 | 1.19 |
| Días con abs(z) > 2 | 8.2% | 23.4% | 28.1% |
| Días con z > +2 | 3.9% | 14.0% | 28.1% |
| Rango máximo de un peso | 0.0109 | 0.0083 | 0.0218 |
| Media estimada μ̂ al inicio y al final | −0.398 a −0.397 | −0.397 a −0.394 | −0.394 a −0.375 |

Lo que muestran:
- En prueba, el spread deja de ser estacionario (ADF p = 0.111) y su vida media se cuadruplica.
- En validación, el spread se queda por encima de su media: todos los días con abs(z) > 2 son del lado positivo. La media estimada sube apenas 0.019 en dos años y medio, porque la ganancia del Filtro 2 es de 0.01. Por eso el z sigue marcando "spread caro" y la regla abre cortos una y otra vez, como se ve en la sección de estrategia.
- La varianza de las innovaciones del Filtro 1 sube a 1.55 en validación. Según S13 p17, el filtro está sobreconfiado: su P es demasiado chica para lo que pasa fuera de muestra.

## Figuras

Figura 3, en `docs/figures/fig3_pesos.png`, con toda la muestra. Los pesos casi no se mueven en entrenamiento (rango máximo 0.0109). Después de 2021 se desplazan un poco más: XEL pasa de 0.411 a 0.371, que es el mayor cambio de toda la muestra (0.040).

Figura 5, en `docs/figures/fig5_ganancias.png`. La ganancia del Filtro 1 baja durante 2014, se estabiliza y salta en marzo de 2020, cuando los precios caen de golpe y cambia `H_t`. Después de 2021 oscila más, con picos cercanos a 0.008 en la media de 21 días. La del Filtro 2 sube muy despacio de 0.0096 a 0.0100 y ahí se queda.

## Nuestra lectura

Lo que nos queda claro es que hay un intercambio directo: con una Q que conserva el spread, los pesos son casi fijos; con una Q que los mueve de verdad, el spread se deshace (a Q/R = 0.01, la vida media es de 0.7 días). Fuera de muestra se vio el costo de ese intercambio. Elegimos una Q chica para no comernos la señal en entrenamiento, y por eso ninguno de los dos filtros se adaptó cuando el spread cambió de nivel después de 2021: los pesos se movieron 0.02 y la media 0.019, mientras el z se quedaba arriba de +2.

El Filtro 1 no rescató la relación, como anticipamos, y el Filtro 2 tampoco corrigió el nivel a tiempo. Su valor aquí fue dar una estimación causal y medir la incertidumbre: la varianza de las innovaciones de 1.55 en validación era una alarma temprana de que el modelo ya no describía los datos. Con eso podría construirse una regla para apagar la estrategia, que es la defensa que proponemos en la sección de estrategia.
