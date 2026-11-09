# Arbitraje estadístico con cointegración y filtros de Kalman

**Laboratorio 03.** Microestructura y Sistemas de Trading, ITESO, otoño 2026.

**Equipo 4:**
- Juan Pablo Barba González (JPBAG0)
- Gian Carlo Campos Sayavedra (GCCS11)
- Ian Carlo Escalante Martínez (ianmartinez01)

**Repositorio:** github.com/GCCS11/Lab03_MyST_Equipo4. Todo se reproduce con `python main.py`. Los pasos que congelan parámetros están en el README.

## Resumen

**El grupo.** Operamos un portafolio reversivo de 6 utilities eléctricas reguladas de EE. UU.: AEP, XEL, ED, EXC, WEC y ES. Lo elegimos solo con entrenamiento (2014 a 2021-09), entre los 210 grupos de 6 que se forman con 10 utilities. En total evaluamos 1,576 grupos.

**El modelo.**
- El Filtro 1 estima los pesos dinámicos del portafolio.
- El VECM confirma la corrección de error, con βᵀα = −0.066 y una vida media de unos 10 días.
- El Filtro 2 estima la media y la varianza de la desviación de equilibrio, y con ellas se calcula un z-score causal.
- La regla entra con abs(z) > 2 y sale con abs(z) < 0.5, con stop en abs(z) > 3 o a los 27 días.
- Los umbrales se eligieron con walk-forward dentro de entrenamiento y se congelaron con un commit antes de ver prueba y validación.

**Resultados con costos** (0.125% por pata y 0.25% anual de préstamo):

| | Entrenamiento | Prueba | Validación |
|---|---|---|---|
| Sharpe | 0.59 | 0.32 | −0.88 |
| Retorno total | 6.4% | 1.0% | −3.6% |
| Drawdown máximo | 2.2% | 1.0% | 4.0% |
| Operaciones | 28 | 6 | 6 |

**Conclusión.** La relación existió entre 2014 y 2021, pero dependía del periodo y se deshace después de 2021. En validación, la vida media del spread pasa de 9 a 64 días, y solo 3% de las ventanas de 2 años conservan la relación. La selección está inflada por las pruebas múltiples: en una prueba anidada dentro de entrenamiento, solo 4% de los candidatos siguen estacionarios en el tramo siguiente. El stop y la tenencia máxima limitaron la pérdida. La defensa que proponemos es volver a probar la relación de forma causal y apagar la estrategia cuando se cae.

## Contenido

1. Datos, selección del grupo, backtest y métricas (figuras 1, 4, 7 y 8).
2. Filtros de Kalman (figuras 3 y 5).
3. Estrategia, resultados y "¿estructural o artefacto?" (figuras 2 y 6).
4. Anexo: formulación de Powell del Filtro 1, del Filtro 2 y del sistema completo.

El uso de asistencia de IA de cada integrante está en el README.
