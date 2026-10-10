# Resultados

Esta carpeta contiene los artefactos y análisis de una corrida anterior sobre el benchmark traducido. Los archivos `*_named.csv` enlazan las etiquetas numéricas a los nombres canónicos de las 77 intenciones.

La comparación global (`model_comparison.csv`) registra accuracy, precision macro, recall macro y F1 macro junto con tiempos de codificación por lotes. En esa corrida, MPNet obtuvo F1 macro de 0.8553; MiniLM 0.8488 y DistilUSE 0.8348.

**Advertencia:** la latencia histórica en `model_comparison.csv` se calculó como tiempo de codificación de test más predicción del clasificador, dividido entre el total de tickets. Es latencia amortizada por lote y no equivale a latencia online. El notebook y `src/benchmark.py` ahora añaden mediciones individuales p50/p95/p99, que deberán completarse al ejecutar el experimento en el entorno objetivo. No sobrescribas el histórico sin guardar la configuración de cada corrida.

No se incluyen copias completas del dataset ni tickets reales. Las salidas de la siguiente ejecución se generan en `results/run_latest/`.
