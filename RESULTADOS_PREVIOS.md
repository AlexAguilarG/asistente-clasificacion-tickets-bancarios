# Revisión de resultados por categoría — Banking77 traducido al español

## Qué se evaluó

Se analizaron los CSV del experimento de Google Colab: 6,036 ejemplos de entrenamiento y 1,925 de prueba, con 25 ejemplos de prueba por cada una de las 77 intenciones. El modelo seleccionado por la regla automática del notebook fue MPNet multilingual v2, porque la regla prioriza F1 macro entre los modelos que cumplen el umbral de latencia.

## Resultados globales

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Latencia media amortizada (ms/ticket) | Memoria GPU pico (MB) |
|---|---:|---:|---:|---:|---:|---:|
| MPNet multilingual v2 | 85.66% | 86.20% | 85.66% | 85.53% | 1.217 | 1287.8 |
| MiniLM-L12-v2 | 84.99% | 85.67% | 84.99% | 84.88% | 0.411 | 568.6 |
| DistilUSE multilingual v2 | 83.74% | 84.08% | 83.74% | 83.48% | 0.687 | 1299.7 |

## Categorías con mayor dificultad para MPNet

Las cinco categorías con menor F1 macro-clase de MPNet fueron:

- `pending_transfer`: F1 0.60, recall 0.56 (14 de 25 casos positivos quedaron sin identificar correctamente).
- `topping_up_by_card`: F1 0.62, recall 0.60.
- `transfer_not_received_by_recipient`: F1 0.63, recall 0.52.
- `direct_debit_payment_not_recognised`: F1 0.68, recall 0.52.
- `declined_card_payment`: F1 0.70, recall 0.76.

Las métricas por clase son sobre 25 ejemplos de prueba cada una; diferencias pequeñas en números absolutos pueden cambiar de forma apreciable el F1. No interpretar el resultado por clase como una estimación robusta de producción.

## Confusiones frecuentes de MPNet

- `transfer_not_received_by_recipient` → `pending_transfer`: 6 casos.
- `get_disposable_virtual_card` → `getting_virtual_card`: 6 casos.
- `beneficiary_not_allowed` → `failed_transfer`: 5 casos.
- `declined_card_payment` → `reverted_card_payment?`: 4 casos.
- `pending_transfer` → `transfer_timing`: 4 casos.
- `pending_transfer` → `balance_not_updated_after_bank_transfer`: 4 casos.

La confusión entre una transferencia no recibida y una transferencia aún pendiente es especialmente relevante para la asignación de cola y la urgencia de atención.

## Categorías sensibles y decisión del modelo

| Intención | MiniLM F1 / Recall | MPNet F1 / Recall | DistilUSE F1 / Recall |
|---|---:|---:|---:|
| `card_payment_not_recognised` | 0.74 / 0.64 | 0.72 / 0.72 | 0.69 / 0.72 |
| `compromised_card` | 0.70 / 0.76 | 0.75 / 0.76 | 0.71 / 0.68 |
| `lost_or_stolen_card` | 0.88 / 0.92 | 0.85 / 0.92 | 0.85 / 0.88 |
| `lost_or_stolen_phone` | 0.98 / 1.00 | 1.00 / 1.00 | 0.98 / 1.00 |
| `failed_transfer` | 0.78 / 0.92 | 0.76 / 0.88 | 0.74 / 0.80 |
| `transfer_not_received_by_recipient` | 0.63 / 0.52 | 0.63 / 0.52 | 0.48 / 0.40 |
| `pending_transfer` | 0.69 / 0.72 | 0.60 / 0.56 | 0.50 / 0.48 |
| `declined_transfer` | 0.83 / 0.76 | 0.83 / 0.76 | 0.83 / 0.76 |
| `transaction_charged_twice` | 0.92 / 0.96 | 0.92 / 0.92 | 0.90 / 0.92 |
| `direct_debit_payment_not_recognised` | 0.70 / 0.56 | 0.68 / 0.52 | 0.63 / 0.52 |

## Recomendación

**No conviene declarar un ganador único sin definir el costo de los distintos errores.** MPNet obtiene el F1 macro global más alto (85.53%) y mejores cifras para `card_payment_not_recognised` en recall (0.72, frente a 0.64 de MiniLM) y para `compromised_card` en F1 (0.75, frente a 0.70 de MiniLM). MiniLM es mucho más eficiente en esta medición (0.411 ms/ticket amortizado frente a 1.217 ms de MPNet; memoria pico 569 MB frente a 1,288 MB) y es mejor en algunas categorías de transferencias como `pending_transfer` (F1 0.69 frente a 0.60) y `failed_transfer` (F1 0.78 frente a 0.76).

Para el prototipo recomendaría **mantener ambos candidatos**: MPNet como referencia de desempeño y MiniLM como alternativa eficiente. La decisión final debe ponderar recall por intención crítica, latencia online real, consumo y errores de negocio. No desplegar automáticamente la clasificación de categorías sensibles; usar revisión humana y reglas de escalamiento.

## Advertencias metodológicas importantes

1. **Corrección de etiquetas:** los CSV iniciales usaban `intent_48` y similares porque el notebook no logró cargar los nombres canónicos de clase. Los archivos con sufijo `_named` mapean las etiquetas numéricas al orden canónico de Banking77 publicado por Hugging Face. La tabla de etiquetas del dataset original confirma la correspondencia entre IDs e intenciones y la ficha de Banking77 traducido conserva esos IDs.
2. **Latencia:** la latencia reportada se calcula con la codificación de lotes de prueba dividida entre el número de ejemplos; es una latencia media amortizada en batching, no una medición de solicitud online individual. Para producción, medir con batch size 1, warm-up, varias repeticiones, sincronización de GPU y percentiles p50/p95/p99.
3. **Validez externa:** Banking77 traducido es un benchmark académico; no equivale a tickets auténticos de clientes mexicanos. Validar con un conjunto local autorizado y anonimizado, manteniendo un conjunto de prueba independiente.
4. **Umbral de confianza:** el umbral 0.70 es inicial y no necesariamente calibrado. Evaluar calibración y curvas de precisión-cobertura antes de usarlo para desviar tickets.
5. **Interpretación de métricas:** cada clase tiene 25 observaciones de test; los números absolutos y métricas por intención son sensibles a unos cuantos errores.

## Fuente para la correspondencia de etiquetas

Hugging Face, `PolyAI/banking77`: https://huggingface.co/datasets/PolyAI/banking77
Hugging Face, `DeepPavlov/banking77-translated`: https://huggingface.co/datasets/DeepPavlov/banking77-translated

Los nombres de intención son la taxonomía original en inglés del dataset; las frases de entrada utilizadas en el experimento fueron traducciones al español.

## Evaluación del umbral de confianza (análisis retrospectivo)

Se evaluó el score máximo de la regresión logística de cada modelo para umbrales entre 0.30 y 0.90. Además de la regla por umbral, se aplicó una **puerta de seguridad analítica**: las predicciones etiquetadas como `card_payment_not_recognised`, `cash_withdrawal_not_recognised`, `direct_debit_payment_not_recognised`, `compromised_card`, `lost_or_stolen_card`, `lost_or_stolen_phone`, `transaction_charged_twice` y `transfer_not_received_by_recipient` permanecen en revisión humana, sin importar el score. Este conjunto es ilustrativo y requiere aprobación de Riesgos/Operación.

Con MPNet, antes de excluir las categorías sensibles, umbral **0.50** acepta 839 de 1,925 tickets (43.6%) con precisión observada de 98.2%; **0.65** acepta 440 (22.9%) con 99.1% de precisión observada; y **0.70** acepta 326 (16.9%) con 99.1% de precisión observada. Al forzar las categorías sensibles a revisión humana, la cobertura de sugerencias cae más; los valores exactos están en `mpnet_threshold_summary.csv` y `confidence_threshold_analysis.csv`.

Estos valores **no demuestran seguridad en producción**: proceden del conjunto traducido, de una sola muestra de prueba, y el score no está calibrado. El ECE de etiqueta principal en diez intervalos se encuentra en `top_label_calibration_summary.csv`; una discrepancia entre confianza y exactitud refuerza la necesidad de calibrar y validar el umbral con datos mexicanos independientes. Los casos aceptados con cero errores observados no implican error real nulo.

## Latencia individual

El benchmark anterior medía el tiempo de codificación en lotes dividido entre el número de tickets. El notebook actualizado añade una prueba con una llamada por ticket después del calentamiento y registra media, p50, p95 y p99 por modelo. La latencia individual no puede inferirse de los tiempos amortizados anteriores y debe obtenerse al ejecutar el nuevo notebook en el entorno objetivo.
