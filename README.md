# Dataset: origen, campos y trazabilidad

## Fuente de obtención

El experimento utiliza el repositorio público **`DeepPavlov/banking77-translated`**, configuración `es`, alojado en Hugging Face Datasets:

- Ficha del conjunto traducido: https://huggingface.co/datasets/DeepPavlov/banking77-translated
- Fuente original BANKING77: https://huggingface.co/datasets/PolyAI/banking77
- Publicación académica original: Casanueva et al. (2020), https://arxiv.org/abs/2003.04807

BANKING77 fue publicado por PolyAI para investigación de clasificación de intenciones y contiene 13,083 consultas de banca digital anotadas con 77 intenciones. La versión `es` de `DeepPavlov/banking77-translated` ofrece una variante en español y conserva el campo `label` como identificador numérico de las intenciones. La ficha visible del conjunto traducido muestra los campos `text`, `label`, `text_gemma` y `text_qwen`.

**Alcance de la procedencia:** la ficha de BANKING77 describe el corpus como consultas de banca en línea, pero no documenta por completo quién produjo las frases originales ni todo el proceso de recopilación. La ficha de la versión traducida tampoco explica suficientemente el procedimiento de traducción, la versión del modelo, los prompts ni una validación humana. Por tanto, no debe afirmarse que son tickets reales de una entidad específica ni que la traducción representa español mexicano.

**Licencia:** la ficha del BANKING77 original declara CC BY 4.0. No se debe asumir automáticamente que esa declaración resuelve la situación legal de la variante traducida; verificar los términos de la versión `DeepPavlov/banking77-translated` antes de redistribuirla o usarla comercialmente.

## Diccionario de datos

| Campo | Tipo en la ficha consultada | Descripción | Uso en el experimento |
|---|---|---|---|
| `text` | string | Enunciado original en inglés. | Trazabilidad y comparación; no alimenta el modelo base. |
| `label` | integer (`int64` en la ficha) | ID numérico de la intención, en el rango 0–76. | Variable objetivo que el clasificador aprende a predecir. Los nombres se corresponden con las 77 clases del BANKING77 original. |
| `text_gemma` | string | Variante traducida al español identificada por el sufijo `gemma`. La ficha no basta para confirmar todos los detalles de generación. | Texto de entrada principal (`TEXT_FIELD = "text_gemma"`). |
| `text_qwen` | string | Variante alternativa en español identificada por el sufijo `qwen`. | No se utiliza en el experimento base; puede usarse para un análisis de sensibilidad. |

La clase `label = 11`, por ejemplo, corresponde a `card_arrival` en el BANKING77 original. Para documentación, el script de inspección exporta la distribución de cada etiqueta y el esquema observado en el snapshot descargado.

## Particiones y tratamiento

Se cargan los splits públicos `train` y `test` por separado. No se mezclan ejemplos entre ellos. Después se aplica limpieza mínima: conversión del texto a cadena, eliminación de espacios al inicio y final, descarte de textos vacíos y etiquetas no numéricas. Se conserva el ID de clase.

Para limitar el tiempo de cómputo y balancear la comparación entre modelos, se toma como máximo:

- 80 ejemplos por intención en entrenamiento.
- 25 ejemplos por intención en prueba.
- Semilla aleatoria fija: `42`.

Por eso, el número final de filas puede ser menor que `77 × 80 = 6,160` si algunas clases del split de entrenamiento contienen menos de 80 registros válidos. En la ejecución previa se obtuvieron 6,036 ejemplos de entrenamiento y 1,925 de prueba; las cantidades exactas deben volver a leerse del manifiesto de cada corrida.

## Cómo documentar y fijar una versión del dataset

La rama `main` puede cambiar. Para que la corrida sea reproducible, se debe registrar el SHA completo del commit del repositorio de datos, además de las versiones del entorno.

```bash
pip install -r requirements.txt
python src/inspect_dataset.py --revision main --output results/dataset_profile
```

El script escribe `results/dataset_profile/dataset_manifest.json`, incluido el campo `resolved_revision_sha`. Copia ese SHA y úsalo tanto al preparar los CSV como al ejecutar la comparación:

```bash
python src/prepare_dataset.py --revision <SHA_COMPLETO> --train-per-class 80 --test-per-class 25 --output data/processed
python src/benchmark.py --revision <SHA_COMPLETO> --train-per-class 80 --test-per-class 25 --batch-size 64 --output results
```

Las salidas incluyen manifiestos con la fuente, el SHA, los parámetros, las cantidades observadas y versiones de las bibliotecas. Las cantidades y las métricas de una ejecución nueva deben tomarse de esos archivos, no de cifras escritas manualmente en el informe.

## Limitaciones para el proyecto bancario

Banking77 traducido es útil como benchmark académico, pero no representa necesariamente la taxonomía, el vocabulario, los procesos ni el español de México de un banco real. Antes de un piloto se necesitaría un conjunto separado de tickets locales, con autorización, anonimización, control de acceso y validación del mapeo de intenciones. No se deben subir tickets reales, identificadores, números de cuenta, tokens o secretos a GitHub ni a Hugging Face.
