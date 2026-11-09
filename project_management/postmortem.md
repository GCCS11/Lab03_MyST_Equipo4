# Postmortem: Lab 03, Equipo 4

Escrito el domingo 8-nov-2026, día de la entrega. Lo redactó JP con lo que quedó en el tablero (`sprint_01.md`, `sprint_02.md`) y en el historial de Git; Gian e Ian pueden agregar su punto de vista abajo.

## Qué entregamos
- **Pipeline:** `python main.py` corre sin internet en unos 3 segundos y escribe todos los `artifacts/`.
- **Pruebas:** las 4 obligatorias más las unitarias, 118 pruebas en verde el domingo en la noche.
- **Proceso:** un grupo de 6 utilities elegido solo con entrenamiento, dos filtros de Kalman, VECM, regla con candado y backtest con costos.
- **Theta:** congelado con commit y tag antes de ver prueba y validación.
- **Visualización:** dashboard en Streamlit con vista ejecutiva y técnica, más una copia estática en `docs/dashboard.html`. Las 8 figuras del reporte.
- **Respuesta a "¿estructural o artefacto?":** se sostiene con cifras propias y con código que las reproduce (`cointegration/diagnostico.py`).
- **Commits:** los tres integrantes tienen commits propios en `main`, todos por PR con merge commit.

## Qué salió bien
- **Decidir pronto y por escrito.** El 4-nov fijamos el grupo de 6, la decisión de no hacer Risk Parity, la escala en log-precios y las fechas fijas. Las propuestas se daban por buenas si nadie objetaba en un plazo, así que nadie se quedó esperando.
- **Contratos entre módulos.** Cada quien desarrolló con datos sintéticos y con las columnas acordadas, así que la integración del viernes y el sábado fue casi directa.
- **Separar la elección de la evaluación.** Las reglas de selección, de los parámetros de los filtros y de theta se escribieron en el código antes de correr. Theta quedó en un commit propio (`4557c17`, tag `theta-congelado`), y los módulos que leen prueba o validación se niegan a correr sin él.
- **Reportar lo que salió, no lo que queríamos.** Validación pierde (−3.6%) y lo explicamos con el diagnóstico en vez de cambiar de grupo.

## Qué salió mal
- **Integración tardía.** Casi todos los PR se fusionaron entre el sábado 7 y el domingo 8. Ian y JP no tuvieron commits en GitHub hasta mitad de semana, y los PR de Gian llegaron en bloque el sábado.
- **Trabajo duplicado.** Gian y JP conectaron `main.py` con el motor en paralelo (PR #12 y PR #13), y uno se cerró sin fusionar. Faltó avisar en el tablero quién lo estaba haciendo.
- **Fechas que no se cumplieron.** El hito de "datos y selección" del jueves 5 se cumplió hasta el sábado, y la corrida única se movió del viernes al sábado en la noche. El reporte se armó el domingo en la noche, no a las 18:00.
- **Ya habíamos visto prueba y validación.** En la exploración del 4-nov el equipo vio resultados de grupos sobre prueba y validación antes de fijar el método. No se usaron para elegir, pero lo declaramos.
- **Preguntas al profesor sin respuesta.** La prueba 4 en datos reales, el "residual del VECM", el cuarto entregable y el grupo de Canvas. Trabajamos con supuestos declarados.

## Qué aprendimos del problema
- **El riesgo más grande era la ruptura fuera de muestra, y ocurrió.** El charter lo puso como riesgo 1 desde el 4-nov. Los pesos dinámicos no lo resuelven: con una Q que conserva el spread, los pesos casi no se mueven.
- **Las pruebas múltiples son reales.** Con 1,576 grupos evaluados, la prueba anidada mostró que pasar el embudo dentro de muestra casi no predice que la relación siga.
- **Los costos dominan.** Las comisiones se llevan cerca de la mitad del P&L bruto. Una estrategia de spread con volatilidad de 1.5% anual deja poco margen.

## Qué haríamos distinto
1. Abrir un PR pequeño cada día desde el lunes, aunque el módulo no esté completo.
2. Asignar en el tablero cada tarea de integración a una sola persona antes de empezarla.
3. Agregar desde el principio una prueba causal de la relación (Johansen o ADF móvil) que apague la estrategia, y elegir sus parámetros en entrenamiento.
4. Separar desde el día 1 la exploración de la evaluación, para que nadie vea prueba ni validación antes de congelar.
5. Hacer las preguntas al profesor por escrito el primer día y registrar la respuesta en el charter.

## Comentarios de Gian e Ian
(Agregar aquí.)
