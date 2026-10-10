# Asistente inteligente para clasificación y priorización de tickets bancarios

## Descripción del proyecto

Este repositorio contiene un prototipo académico de procesamiento de lenguaje natural (PLN) para **clasificar consultas de banca digital por intención**. El trabajo forma parte del proyecto *Asistente inteligente para clasificación y priorización de tickets de servicio al cliente*. Evalúa tres modelos de embeddings multilingües de Hugging Face y compara sus resultados usando un mismo clasificador `LogisticRegression`.

El objetivo de esta fase es producir evidencia reproducible para seleccionar candidatos, identificar errores por intención y definir qué casos deberían pasar a revisión humana. El prototipo **no ejecuta operaciones bancarias, no resuelve solicitudes por sí mismo y no está validado para producción**.

## Dataset y procedencia

Se utiliza `DeepPavlov/banking77-translated`, configuración `es`, disponible en [Hugging Face Datasets](https://huggingface.co/datasets/DeepPavlov/banking77-translated). La referencia original es [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77), un benchmark de 13,083 consultas y 77 intenciones de banca digital. La variante española presenta los campos `text`, `label`, `text_gemma` y `text_qwen`; la configuración base utiliza `text_gemma` como texto y `label` como clase.

La traducción publicada no representa necesariamente el español de México ni los tickets de una institución concreta. La ficha del conjunto traducido no documenta por completo el proceso de traducción; se debe verificar la licencia y los términos de esa variante antes de redistribuirla o usarla comercialmente. Los datos se descargan de Hugging Face al ejecutar los scripts; no se versionan tickets ni copias completas del dataset en este repositorio.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── README.md
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── docs/
│   ├── ORIGEN_ESTRUCTURA_REPRODUCIBILIDAD.md
│   └── ORIGEN_ESTRUCTURA_REPRODUCIBILIDAD.docx
├── notebooks/
│   └── Comparacion_modelos_HF_Banking77_es_LATENCIA_CONFIANZA.ipynb
├── results/
│   ├── README.md
│   ├── RESULTADOS_PREVIOS.md
│   ├── model_comparison.csv
│   ├── confidence_threshold_analysis.csv
│   ├── classification_report_*_named.csv
│   └── gráficas y tablas de análisis por intención
└── src/
    ├── inspect_dataset.py
    ├── prepare_dataset.py
    └── benchmark.py
```

## Entorno de ejecución

**Recomendado:** Python 3.10 o 3.11, entorno virtual y acceso a internet para descargar los datasets y modelos. GPU compatible con PyTorch (por ejemplo, Google Colab con GPU T4) recomendada para acelerar la comparación; también se puede ejecutar en CPU, con mayor tiempo de procesamiento.

Las dependencias principales se encuentran en `requirements.txt`. El manifiesto de cada ejecución guarda las versiones reales de Python y de librerías, la revisión exacta del dataset, las dimensiones de la muestra y el dispositivo utilizado.

### Instalación local

En Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si se requiere una versión específica de PyTorch con CUDA, instalarla según la guía oficial correspondiente al sistema y GPU: https://pytorch.org/get-started/locally/ . Colab suele traer un entorno de PyTorch preconfigurado.

## Pasos de ejecución reproducible

Ejecuta todos los comandos desde la raíz del repositorio.

### 1. Inspeccionar el conjunto y fijar la revisión

```bash
python src/inspect_dataset.py --revision main --output results/dataset_profile
```

Este paso genera `results/dataset_profile/dataset_manifest.json`, con el SHA del commit efectivamente descargado. Copia ese `resolved_revision_sha` y úsalo en los siguientes pasos, para no depender de los cambios futuros en la rama `main`.

### 2. Preparar la muestra balanceada

```bash
python src/prepare_dataset.py --revision <SHA_COMPLETO> --train-per-class 80 --test-per-class 25 --output data/processed
```

Se conservan las particiones originales `train` y `test`, se eliminan textos vacíos y etiquetas inválidas y se toma una muestra por clase con semilla `42`. No se mezclan ejemplos entre entrenamiento y prueba. La cantidad final puede ser menor a 6,160 registros de entrenamiento si una clase no contiene 80 ejemplos válidos.

### 3. Ejecutar el benchmark

```bash
python src/benchmark.py --revision <SHA_COMPLETO> --train-per-class 80 --test-per-class 25 --batch-size 64 --latency-sample-size 300 --warmup-samples 10 --output results/run_latest
```

El script descarga los tres modelos, codifica los textos, entrena una regresión logística por modelo y genera métricas globales, reportes por clase, matrices de confusión, pares de confusión, predicciones locales, análisis exploratorio de umbrales y un manifiesto técnico.

- El tiempo amortizado por lote es el tiempo de codificación de la partición dividido entre su número de registros.
- La latencia individual p50/p95/p99 se mide con una llamada por ticket (batch size 1), después de calentamiento, incluyendo codificador y clasificador. No incluye carga inicial del modelo, red, colas ni latencia de una aplicación completa.
- Los umbrales de confianza de la regresión logística son exploratorios y no deben considerarse probabilidades calibradas.

### 4. Ejecutar desde Google Colab

1. Sube el repositorio a GitHub o carga el notebook `notebooks/Comparacion_modelos_HF_Banking77_es_LATENCIA_CONFIANZA.ipynb` en Colab.
2. En **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, selecciona GPU cuando esté disponible.
3. Ejecuta todas las celdas en orden desde el principio.
4. Consulta los CSV y gráficos generados en `results/`.
5. Registra la fecha, la GPU, las versiones de librerías y la revisión del dataset. No incluyas tokens en el código; los recursos públicos no requieren autenticación en un uso habitual.

## Resultados previos y conclusiones

La ejecución previa de comparación por lotes se hizo en Google Colab con una GPU NVIDIA Tesla T4, 6,036 ejemplos de entrenamiento y 1,925 de prueba. Los valores siguientes son resultados del benchmark guardado; **las latencias mostradas son amortizadas por lotes, no la latencia individual p50/p95/p99**.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Latencia por lote amortizada | Memoria GPU pico aprox. |
|---|---:|---:|---:|---:|---:|---:|
| MPNet multilingual v2 | 85.66% | 86.20% | 85.66% | **85.53%** | 1.217 ms/ticket | 1,288 MB |
| MiniLM-L12-v2 | 84.99% | 85.67% | 84.99% | 84.88% | **0.411 ms/ticket** | **569 MB** |
| DistilUSE multilingual v2 | 83.74% | 84.08% | 83.74% | 83.48% | 0.687 ms/ticket | 1,300 MB |

**Conclusión técnica:** MPNet obtiene el F1 macro global más alto en esta muestra; MiniLM ofrece una alternativa más ligera y con menor latencia amortizada. Por ello, conviene mantener ambos como candidatos: MPNet como referencia de desempeño y MiniLM como candidato de eficiencia. La selección definitiva requiere medir latencia individual en el entorno de destino y definir con Operación/Riesgos el costo de los errores por intención.

Entre las categorías difíciles para MPNet se encuentran `pending_transfer`, `topping_up_by_card` y `transfer_not_received_by_recipient`. La confusión entre una transferencia no recibida y una transferencia pendiente puede afectar la priorización; no basta la exactitud global para decidir la operación.

La exploración retrospectiva del umbral de confianza indica el intercambio entre cobertura y revisión humana. Con MPNet, un umbral de 0.65 aceptó 401 de 1,925 ejemplos (20.8%) con precisión observada de 99.0% antes de aplicar reglas que fuerzan algunas intenciones sensibles a revisión. Esto es un resultado de una muestra traducida, no una garantía de cero riesgo ni una regla validada para producción.

## Limitaciones, privacidad y uso responsable

- No emplear tickets reales sin autorización, base legal, minimización de datos, anonimización y control de acceso.
- No subir datos de clientes, números de cuenta, nombres, teléfonos, tokens ni secretos a GitHub o Hugging Face.
- `BANKING77` traducido es un benchmark académico y no sustituye una evaluación con datos representativos del español mexicano.
- Cada intención de la muestra de prueba tiene hasta 25 ejemplos, por lo que las métricas por clase son sensibles a pocos errores.
- El umbral no está calibrado. Antes de sugerir una cola de forma automática, se deben validar los resultados en un conjunto independiente de tickets locales y definir una lista formal de casos sensibles que siempre pasan a revisión humana.
- El prototipo genera sugerencias de intención: no toma decisiones financieras ni realiza acciones en cuentas.

## Referencias

- Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulic, I. (2020). *Efficient intent detection with dual sentence encoders*. https://arxiv.org/abs/2003.04807
- DeepPavlov. (s. f.). *banking77-translated* [Conjunto de datos]. Hugging Face. https://huggingface.co/datasets/DeepPavlov/banking77-translated
- PolyAI. (s. f.). *BANKING77* [Conjunto de datos]. Hugging Face. https://huggingface.co/datasets/PolyAI/banking77
- Hugging Face. (s. f.). *Sentence Transformers*. https://huggingface.co/sentence-transformers
- Tecmilenio. (2026). *Tema 9. Selección de modelos preentrenados* [Material de curso].
