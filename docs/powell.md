# Formulación de Powell: Filtro 1, Filtro 2 y sistema completo

Lab 03, Equipo 4. Sección 3.5 del enunciado. Todas las cifras salen de `artifacts/filtros.json`, sobre el grupo AEP, XEL, ED, EXC, WEC y ES, y usan solo entrenamiento (2014-01-02 a 2021-09-08).

## Cómo se relacionan los dos listados de pasos

El enunciado y la lámina S11 p15 nombran distinto los 6 pasos. Seguimos el enunciado y dejamos la equivalencia:

| Paso | Enunciado (3.5) | S11 p15 |
|---|---|---|
| 1 | Definir el problema | Narrativa |
| 2 | Modelar el estado | Elementos centrales |
| 3 | Modelar la transición | Modelo matemático |
| 4 | Modelar la observación | Modelo de incertidumbre |
| 5 | Definir la regla de decisión | Diseño de políticas |
| 6 | Definir el aprendizaje o actualización | Evaluación de políticas |

Notación: `P_t` son log-precios, `y_t` es la pata normalizada (AEP), `x_t` son las otras 5 patas y `φ` es la persistencia del error de equilibrio.

## 1. Filtro 1: pesos dinámicos

### Los 6 pasos

1. Problema. Los pesos del portafolio reversivo pueden cambiar con el tiempo. Queremos estimarlos en cada fecha usando solo lo conocido hasta ese momento, sin que el filtro absorba la desviación que la estrategia quiere operar.
2. Estado. `θ_t`, los pesos de las 5 patas no normalizadas, y su covarianza `P_t`. AEP queda fija en +1 (S14 p13), así que `β_t = (1, θ_t)` y el signo y la escala no cambian entre fechas.
3. Transición. Paseo aleatorio: `θ_t = θ_{t-1} + w_t`, con `w_t ~ N(0, Q·I)`. En la predicción, `P⁻_t = P_{t-1} + Q·I`.
4. Observación. `y_t − c = −x_tᵀθ_t + u_t`, con `u_t = φ·u_{t-1} + e_t` y `e_t ~ N(0, R)`. Como `u_t` es persistente, se cuasi-diferencia: `ỹ_t = y_t − φ·y_{t-1} − (1−φ)·c = H_t·θ_t + e_t`, con `H_t = −(x_t − φ·x_{t-1})ᵀ`.
5. Regla de decisión. La corrección `x_t = K_t·ν_t`, con `ν_t = ỹ_t − H_t·θ⁻_t`, `F_t = H_t·P⁻_t·H_tᵀ + R` y `K_t = P⁻_t·H_tᵀ / F_t`. Es una PFA con `θ = (Q, R)` (S13 p23).
6. Aprendizaje. `θ_t = θ⁻_t + K_t·ν_t` y `P_t = (I − K_t·H_t)·P⁻_t`. Los parámetros se eligen con walk-forward dentro de entrenamiento y después quedan fijos.

### Los 5 elementos

| Elemento | Definición |
|---|---|
| Variables de estado | Información `I_t = (P_t, P_{t-1})`; creencia `B_t = (θ̂_t, P_t)`. No tiene estado físico |
| Variable de decisión | `x_t = K_t·ν_t`, la corrección a los pesos |
| Información exógena | `W_{t+1} = P_{t+1}`, los precios del día siguiente |
| Función de transición | Las ecuaciones de predicción y actualización de los pasos 3 y 6 |
| Función objetivo | `min E[ν_t²]` (S13 p23); `Q` se elige por vida media y Sharpe neto en walk-forward |

El spread de `t` usa los pesos previos: `S_t = y_t + x_tᵀθ⁻_t`. Son conocidos antes de ver los precios de `t`, y esto evita el look-ahead.

### Ganancia de Kalman y cómo leerla

`K_t` dice cuánto mueve al peso una sorpresa de una unidad. Es grande si los pesos son inciertos y chica si la observación es ruidosa (S13 p12). Aquí la mediana de `|K|` para XEL es 0.0048: una innovación de 1% mueve ese peso apenas 0.00005. `K` es negativa porque `H_t` lleva signo negativo: si AEP sube más de lo esperado, los pesos de las otras patas bajan.

### Justificación de Q y R

- `R = 6.42×10⁻⁵` es la varianza del residual del AR(1) del spread en entrenamiento: un ruido diario de 0.80%.
- `φ = 0.925` sale de ese mismo AR(1), y `c = −0.399` es su media.
- `Q = 6.42×10⁻⁹`, es decir, `Q/R = 0.0001`: cada peso se mueve del orden de 0.00008 al día. La ganancia depende solo de esa razón (S13 p16).
- `P0 = 1.00×10⁻⁵` sale de una pasada previa sobre entrenamiento. Con `P0 = Q`, `P_t` tardaba 1,724 días en acercarse a su nivel, y el burn-in de 60 días no cumplía su función (S13 p16).

Sensibilidad a `Q/R`:

| Q/R | Vida media del spread | Rango máximo de un peso | Mediana de abs(K) en XEL |
|---|---|---|---|
| 0 | 8.9 días | 0.0000 | 0.00000 |
| 0.000001 | 8.7 días | 0.0003 | 0.00047 |
| 0.00001 | 8.2 días | 0.0013 | 0.00161 |
| 0.0001 (elegida) | 6.9 días | 0.0109 | 0.00484 |
| 0.001 | 3.7 días | 0.0880 | 0.01484 |
| 0.01 | 0.7 días | 0.4123 | 0.05075 |

Subir `Q` hace que los pesos persigan el spread: la vida media se desploma y el ADF sale estacionario de forma mecánica. Por eso la verosimilitud no basta como criterio, ya que siempre premia una `Q` mayor.

### Ejemplo numérico

| Fecha | β̂ XEL previo | S_t | ν_t | F_t | ν_t/√F_t | K XEL | P XEL |
|---|---|---|---|---|---|---|---|
| 2014-06-26 | 0.410891 | −0.377218 | −0.005293 | 6.4754×10⁻⁵ | −0.658 | −0.006646 | 9.2382×10⁻⁶ |
| 2014-06-27 | 0.410926 | −0.373840 | 0.004442 | 6.4750×10⁻⁵ | 0.552 | −0.006775 | 9.2416×10⁻⁶ |
| 2014-06-30 | 0.410896 | −0.370295 | 0.005902 | 6.4760×10⁻⁵ | 0.733 | −0.007588 | 9.2443×10⁻⁶ |
| 2014-07-01 | 0.410851 | −0.364105 | 0.008974 | 6.4680×10⁻⁵ | 1.116 | −0.007109 | 9.2475×10⁻⁶ |
| 2014-07-02 | 0.410787 | −0.361894 | 0.005716 | 6.4642×10⁻⁵ | 0.711 | −0.006420 | 9.2512×10⁻⁶ |
| 2014-07-03 | 0.410750 | −0.357068 | 0.008146 | 6.4688×10⁻⁵ | 1.013 | −0.007560 | 9.2539×10⁻⁶ |

Comprobación del 1 de julio: `0.410851 + (−0.007109)(0.008974) = 0.410787`, el peso previo del día siguiente. `F_t` casi no cambia porque `R` domina, y `P` sube poco a poco porque `Q` se suma en cada predicción. El caso escalar de S13 p13 está reproducido en `tests/test_filtro.py`.

## 2. Filtro 2: media y varianza del spread

### Los 6 pasos

1. Problema. Saber en tiempo real qué tan lejos está el spread de su equilibrio, en desviaciones estándar, sin usar el dato del día para juzgar ese mismo dato.
2. Estado. La media latente `μ_t`, su varianza `p_t` y la varianza de la innovación `σ²_ν,t`.
3. Transición. `μ_t = μ_{t-1} + w_t`, con `w_t ~ N(0, Q)`, y `p⁻_t = p_{t-1} + Q`. La escala sigue `σ²_ν,t+1 = λ·σ²_ν,t + (1−λ)·ν_t²` (S13 p20).
4. Observación. `S_t = μ_t + d_t`, con `d_t = φ·d_{t-1} + e_t` y `e_t ~ N(0, R)`. Cuasi-diferenciada: `S_t − φ·S_{t-1} = H·μ_t + e_t`, con `H = 1 − φ`. Con `φ = 0` es el nivel local de S13 p19.
5. Regla de decisión. La corrección `x_t = K_t·ν_t`, con `ν_t = S_t − φ·S_{t-1} − H·μ⁻_t` y `K_t = p⁻_t·H / (H²·p⁻_t + R)`. La salida es `z_t = (S_t − μ⁻_t) / σ_t`, con `σ²_t = σ²_ν,t / (1 − φ²)`.
6. Aprendizaje. `μ_t = μ⁻_t + K_t·ν_t`, `p_t = (1 − K_t·H)·p⁻_t` y la actualización de la escala. Los parámetros se fijan en entrenamiento.

### Los 5 elementos

| Elemento | Definición |
|---|---|
| Variables de estado | Información `I_t = (S_t, S_{t-1})`; creencia `B_t = (μ̂_t, p_t, σ̂²_ν,t)` |
| Variable de decisión | `x_t = K_t·ν_t`, la corrección a la media |
| Información exógena | `W_{t+1} = S_{t+1}`, el siguiente spread |
| Función de transición | Las ecuaciones de los pasos 3 y 6 |
| Función objetivo | `min E[ν_t²]`; `Q`, `φ` y `λ` se eligen por Sharpe neto en walk-forward |

### Ganancia de Kalman y por qué z usa el prior

`K` va de 0.0096 a 0.0100, y además la corrección se aplica con `H = 0.096`: la media casi no reacciona a un día. Si `K` fuera alta, `μ̂` seguiría al spread y `z` se quedaría sin señal.

El z se calcula con el prior porque `μ̂_t` ya absorbió `S_t`. En el nivel local, `z_post = (1 − K_t)·z_prior` (S13 p20), siempre más chico en valor absoluto. Con `φ > 0` la relación es `z_post = z_prior − K_t·ν_t/σ_t`. Ejemplo: el 1 de julio de 2014, `z_prior = 2.146` y `z_post = 2.140`.

### Justificación de Q, R, φ y λ

- `R = 6.53×10⁻⁵` y `φ = 0.904` salen del AR(1) del spread del Filtro 1 en entrenamiento. `μ0 = −0.397` es su media de largo plazo.
- `Q/R = 0.0001` y `λ = 0.99` salieron del walk-forward.
- `P0 = 6.47×10⁻⁶`, de una pasada previa. Al terminar el burn-in, `p_t` ya está en 96% de su nivel final.

El enunciado y S13 p19 parten de un nivel local (`φ = 0`). Lo comparamos con la misma `Q/R` y la misma `λ`:

| Filtro 2 | Varianza innovaciones | Autocorrelación | Ljung-Box p | Autocorrelación de z | Días con abs(z) > 2 |
|---|---|---|---|---|---|
| Nivel local (φ = 0) | 1.037 | 0.903 | 0.000 | 0.903 | 4.7% |
| Con φ = 0.904 (elegido) | 1.064 | −0.087 | 0.000 | 0.926 | 8.2% |

Con `φ` la señal conserva su persistencia y la autocorrelación de las innovaciones casi desaparece, aunque el Ljung-Box sigue rechazando.

### Ejemplo numérico

| Fecha | S_t | μ̂ previa | σ̂ previa | z_t | ν_t | σ²_ν | K_t | p_t |
|---|---|---|---|---|---|---|---|---|
| 2014-06-27 | −0.373840 | −0.397364 | 0.015496 | 1.518 | 0.005315 | 4.3946×10⁻⁵ | 0.009620 | 6.5360×10⁻⁶ |
| 2014-06-30 | −0.370295 | −0.397313 | 0.015468 | 1.747 | 0.005801 | 4.3789×10⁻⁵ | 0.009620 | 6.5365×10⁻⁶ |
| 2014-07-01 | −0.364105 | −0.397257 | 0.015450 | 2.146 | 0.008781 | 4.3688×10⁻⁵ | 0.009621 | 6.5369×10⁻⁶ |
| 2014-07-02 | −0.361894 | −0.397172 | 0.015509 | 2.275 | 0.005390 | 4.4022×10⁻⁵ | 0.009622 | 6.5374×10⁻⁶ |
| 2014-07-03 | −0.357068 | −0.397121 | 0.015483 | 2.587 | 0.008212 | 4.3872×10⁻⁵ | 0.009622 | 6.5379×10⁻⁶ |
| 2014-07-07 | −0.364157 | −0.397042 | 0.015524 | 2.118 | −0.003247 | 4.4108×10⁻⁵ | 0.009623 | 6.5384×10⁻⁶ |

Comprobación del 1 de julio: `z = (−0.364105 + 0.397257)/0.015450 = 2.146`, `μ̂` siguiente `= −0.397257 + 0.009621·0.008781 = −0.397172` y `σ²_ν` siguiente `= 0.99·4.3688×10⁻⁵ + 0.01·0.008781² = 4.4022×10⁻⁵`. La tabla de S13 p21 está reproducida en `tests/test_filtro.py`.

## 3. Sistema completo

### Los 6 pasos

1. Problema. Operar un portafolio de 6 acciones que revierte a su equilibrio, con 80% del capital, pagando comisión y préstamo, y sin usar información futura.
2. Estado. Se separa en físico, información y creencia (tabla siguiente).
3. Transición. Filtro 1, luego Filtro 2, luego la regla. Después, efectivo y posiciones cambian con la operación y los costos.
4. Observación. Cada cierre llega `P_t`. Con él se calculan `S_t` y `z_t`.
5. Regla de decisión. La regla de bandas de S12 p20 con candado después de un stop o de `h_max`, más el tamaño de S14 p20. Es una PFA con `θ = (z_in, z_out, z_stop, h_max)`.
6. Aprendizaje. Las creencias se actualizan cada día. `θ` y los parámetros de los filtros se congelan en entrenamiento.

### Los 5 elementos

Estado:

| Componente | Variables |
|---|---|
| Físico `R_t` | Efectivo, acciones por activo `q_i,t`, periodos abiertos `h_t`, candado |
| Información `I_t` | Precios `P_t` y `P_{t-1}`, spread `S_t` |
| Creencia `B_t` | `θ̂_t`, `P_t` (Filtro 1); `μ̂_t`, `p_t`, `σ̂²_ν,t` (Filtro 2) |

El estado inicial `S_0` contiene lo fijo durante la operación: `β` de Johansen, `Q`, `R`, `φ`, `λ`, `c`, `θ` y los costos.

| Elemento | Definición |
|---|---|
| Variable de decisión | Dirección `d_t ∈ {−1, 0, +1}` y acciones `q_i,t = d_t·sgn(β_i,t)·N_i,t/P_i,t`, con `N_i,t = 0.8·V_t·abs(β_i,t)/Σ abs(β_j,t)` y `P_i,t` el precio real |
| Información exógena | `W_{t+1}`: precios del cierre siguiente |
| Función de transición | Filtros, regla y `efectivo_{t+1} = efectivo_t − Σ Δq_i·P_i,t+1 − comisión − préstamo`. Señal al cierre de `t`, operación al cierre de `t+1` |
| Función objetivo | `max E[Σ C_t]`, con `C_t` el P&L neto de comisión de 0.125% por pata y préstamo de 0.25% anual sobre cortos; `θ` se evalúa por Sharpe neto en walk-forward |

Según S11 p11, una regla que usara `W_{t+1}` no sería una política, porque usaría el futuro. Por eso el z usa el prior y la operación ocurre en `t+1`.

### Ejemplo numérico

Con umbrales ilustrativos `z_in = 2`, `z_out = 0.5`, `z_stop = 3.5` y `h_max = 14`, y un capital de $1,000,000:

| Fecha | z_t | Lectura | d_t | h_t |
|---|---|---|---|---|
| 2014-06-30 | 1.747 | Dentro de la banda, plano | 0 | 0 |
| 2014-07-01 | 2.146 | Cruza z_in: spread caro, se vende | −1 | 0 |
| 2014-07-02 | 2.275 | Se mantiene | −1 | 1 |
| 2014-07-03 | 2.587 | Se mantiene | −1 | 2 |
| 2014-07-07 | 2.118 | Se mantiene | −1 | 3 |

Tamaño objetivo calculado con los pesos y precios del 1 de julio (la operación se ejecutaría al cierre del día siguiente):

| Activo | β̂ | Precio | Nocional | Acciones |
|---|---|---|---|---|
| AEP | 1.0000 | 35.83 | 271,638 | −7,580.36 |
| XEL | 0.4109 | 21.74 | 111,603 | −5,132.39 |
| ED | −0.4113 | 36.53 | 111,727 | 3,058.23 |
| EXC | −0.1872 | 16.46 | 50,848 | 3,089.88 |
| WEC | −0.3375 | 31.11 | 91,691 | 2,946.97 |
| ES | −0.5982 | 30.45 | 162,493 | 5,335.70 |

Los nocionales suman $800,000, el 80% del capital.
