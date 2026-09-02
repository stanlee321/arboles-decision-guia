# Datos de entrada

`riesgo_clinico_simulado.csv` contiene observaciones totalmente sinteticas.
Cada fila representa un caso pedagogico, no una persona real.

| Variable | Tipo | Descripcion |
|---|---|---|
| `id` | entero | identificador artificial |
| `edad` | entero | edad simulada, en anos |
| `presion_sistolica` | continuo | presion simulada, mmHg |
| `colesterol` | continuo | colesterol simulado, mg/dL |
| `fumador` | binaria | 1 si el caso fue marcado como fumador |
| `actividad_horas_semana` | continuo | horas simuladas de actividad semanal |
| `antecedente_familiar` | binaria | indicador familiar simulado |
| `riesgo_alto` | binaria | desenlace simulado que predice el arbol |

El archivo se regenera con `python scripts/generate_data.py`. La relacion entre
variables y desenlace incluye ruido aleatorio para que el ejercicio no sea
perfectamente separable.
