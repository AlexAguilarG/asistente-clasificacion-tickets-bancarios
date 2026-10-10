# Paso 3. Documentación del origen, estructura y reproducibilidad del dataset

**Proyecto:** asistente inteligente para clasificación y priorización de tickets de servicio al cliente bancario  
**Técnica:** procesamiento de lenguaje natural (PLN), clasificación multiclase de intenciones  
**Fuente seleccionada:** `DeepPavlov/banking77-translated`, configuración `es`  
**Revisión fijada para reproducibilidad:** `48246f6c99b06096d9dbd7fcbaf273c6d82df2f0`

## 3.1. Descripción del origen del dataset

Para el experimento se utiliza la versión en español de **BANKING77**, publicada en Hugging Face como `DeepPavlov/banking77-translated`. El corpus original BANKING77 fue presentado por Casanueva et al. (2020) para estudiar la detección de intenciones en consultas de banca digital. Contiene 13,083 consultas en inglés distribuidas en 77 intenciones específicas, por ejemplo, activación de tarjeta, transferencias pendientes, cargos no reconocidos y recepción de dinero. La versión original de PolyAI publica 10,003 registros para entrenamiento y 3,080 para prueba y declara licencia Creative Commons Attribution 4.0 International (CC BY 4.0).

Para este proyecto se seleccionó la configuración `es` de la versión traducida. La revisión consultada registra **9,993 filas de entrenamiento y 3,076 de prueba**, para un total de **13,069 registros**. Esto es 14 registros menos que el corpus original; la ficha pública no explica la causa de esa diferencia, por lo que no se atribuye a una causa concreta. La versión española contiene cuatro columnas: `text`, `label`, `text_gemma` y `text_qwen`.

El experimento usa `text_gemma` como texto de entrada, conservando el campo `label` como categoría objetivo. El nombre de `text_gemma` sugiere una variante asociada a Gemma y `text_qwen` otra variante asociada a Qwen; sin embargo, la ficha pública del repositorio traducido no documenta con suficiente detalle el proceso, las versiones de los modelos, los prompts ni una validación humana de las traducciones. Por rigor, esas denominaciones se describen como nombres de campos, no como una procedencia técnica confirmada. El proceso reproducible de este proyecto consiste en **descargar una revisión identificada del dataset ya publicado y prepararla para el experimento**; no en recrear las traducciones originales.

**Justificación de selección.** BANKING77 es pertinente como benchmark académico porque se limita al dominio bancario y distingue 77 tipos de consulta. La versión en español permite evaluar representaciones multilingües con mensajes comprensibles en español, manteniendo las etiquetas de intención del corpus original. Es apropiada para comparar modelos preentrenados bajo condiciones controladas, pero no debe describirse como un histórico real del banco ni como una muestra representativa del español bancario mexicano.

**Licencia y uso.** La ficha del corpus original PolyAI declara CC BY 4.0. En cambio, la ficha disponible de `DeepPavlov/banking77-translated` no presenta claramente una licencia propia ni una documentación completa de la derivación. Antes de distribuir el dataset traducido o usarlo en un contexto comercial, deben verificarse los permisos y condiciones aplicables a esa versión específica.

## 3.2. Estructura y descripción de los campos

| Campo | Tipo publicado | Descripción y uso en el experimento |
|---|---|---|
| `text` | Texto (`string`) | Consulta original en inglés. Se conserva como referencia de la frase de origen; no es el campo principal de entrada del experimento en español. |
| `label` | Entero (`int64`) | Identificador de la intención, con valores entre 0 y 76. Es la variable objetivo que el clasificador debe predecir. Cada identificador corresponde a una de las 77 categorías canónicas de BANKING77. |
| `text_gemma` | Texto (`string`) | Variante de la consulta en español. Es el campo seleccionado como entrada principal (`TEXT_FIELD = "text_gemma"`). El nombre del campo por sí solo no permite verificar cómo se generó la traducción. |
| `text_qwen` | Texto (`string`) | Segunda variante en español disponible en el dataset. No se usa como entrada principal en el experimento base; puede emplearse en una evaluación de robustez comparando las dos variantes. La ficha pública tampoco describe suficientemente su proceso de generación. |

El objetivo del modelo es aprender la relación entre el texto en español y el identificador `label`. Los valores de `label` son números categóricos: aunque se representen como enteros, no tienen un significado ordinal (por ejemplo, la intención 20 no es “mayor” que la intención 10).

### Particiones y muestreo del experimento

Se respetan las particiones `train` y `test` proporcionadas por la fuente. No se combinan antes de la selección de ejemplos. Para limitar el tiempo de cómputo se aplica una muestra balanceada por clase con semilla fija (`SEED = 42`): hasta 80 registros por intención para entrenamiento y hasta 25 por intención para prueba. El total efectivo puede ser menor si alguna clase no dispone de ese número de registros después de la limpieza. En la ejecución analizada se utilizaron 6,036 registros de entrenamiento y 1,925 de prueba; la muestra de prueba equivale a 25 ejemplos por cada una de las 77 intenciones. Con ese soporte, un cambio de una observación altera el recall de una clase en cuatro puntos porcentuales, por lo que las métricas individuales por intención deben interpretarse con cautela.

La limpieza aplicada al campo de texto consiste en convertirlo a texto, eliminar espacios exteriores y retirar registros cuyo texto o etiqueta estén vacíos. La etiqueta se transforma a entero. No se traduce, reescribe ni parafrasea el contenido durante esta etapa.

## 3.3. Scripts reproducibles y procedimiento de obtención

### Fuente y revisión identificable

- Repositorio: <https://huggingface.co/datasets/DeepPavlov/banking77-translated>
- Configuración: `es`
- Revisión inmutable utilizada: `48246f6c99b06096d9dbd7fcbaf273c6d82df2f0`
- URL de la revisión: <https://huggingface.co/datasets/DeepPavlov/banking77-translated/tree/48246f6c99b06096d9dbd7fcbaf273c6d82df2f0>
- Corpus de referencia: <https://huggingface.co/datasets/PolyAI/banking77>

Fijar la revisión reduce el riesgo de que una modificación futura de la rama principal cambie silenciosamente los archivos recibidos. El script `descarga_prepara_banking77_es.py` realiza las operaciones siguientes:

1. Descarga la configuración `es` de la revisión indicada mediante la biblioteca `datasets`.
2. Comprueba que existan las particiones `train` y `test` y las cuatro columnas esperadas.
3. Exporta los datos originales de cada partición a `train_raw.csv` y `test_raw.csv`.
4. Genera `class_counts_by_split.csv`, un diccionario de campos y `dataset_metadata.json`, con tamaños, valores faltantes, distribución de clases, versiones de paquetes, huellas digitales de las particiones y hashes SHA-256 de los archivos exportados.
5. Limpia el campo `text_gemma` y selecciona, por separado en cada partición, hasta 80 registros por clase para entrenamiento y hasta 25 para prueba, con semilla 42.
6. Guarda `train_experiment.csv` y `test_experiment.csv` para su uso posterior. El script valida que las clases presentes en ambas muestras coincidan.

### Ejecución

En Google Colab o en un entorno Python limpio:

```bash
pip install -r requirements_banking77_es.txt
python descarga_prepara_banking77_es.py
```

El script no requiere token de Hugging Face para este repositorio público. Para repetir la comparación completa de modelos, se utiliza el notebook `Comparacion_modelos_HF_Banking77_es_LATENCIA_CONFIANZA.ipynb`, que vuelve a descargar el mismo dataset y realiza el muestreo con la misma semilla. Las salidas generadas por el script se escriben en `data_banking77_es/`.

### Archivos producidos por el script

| Archivo | Contenido |
|---|---|
| `train_raw.csv`, `test_raw.csv` | Particiones completas descargadas, con las cuatro columnas originales. |
| `class_counts_by_split.csv` | Número de registros por identificador de intención para cada partición. |
| `data_dictionary.csv` | Nombre, tipo detectado y descripción de cada campo. |
| `dataset_metadata.json` | Identificador y revisión de la fuente, información de las particiones, versiones de software, metadatos y hashes. |
| `train_experiment.csv`, `test_experiment.csv` | Muestras limpias y limitadas por clase usando `text_gemma` y `label`. |

## 3.4. Limitaciones y controles de calidad

1. **Proveniencia de traducción incompleta.** La ficha del dataset traducido no describe suficientemente el modelo y procedimiento usados para generar cada variante ni si existió revisión humana. Por ello, no se puede afirmar que las traducciones se regeneraron de forma idéntica a partir de una receta publicada.
2. **Generalización geográfica.** El español de la versión traducida no está documentado como español mexicano. Puede contener expresiones poco naturales o términos que no coincidan con el lenguaje de los clientes de un banco en México.
3. **Diferencia de tamaño.** La versión española declarada contiene 13,069 registros, frente a 13,083 de la fuente original. Debe conservarse el recuento efectivo que imprima el script, sin asumir que ambas versiones son idénticas fila por fila.
4. **Etiquetas y errores.** La etiqueta se hereda del corpus original. Las traducciones podrían cambiar matices semánticos y, en ciertos casos, no expresar con precisión la intención original; se recomienda revisar muestras, especialmente de las categorías de mayor riesgo.
5. **Uso no productivo.** Los resultados solo permiten comparar candidatos en un benchmark académico. Antes de un piloto bancario se requiere un conjunto de tickets reales, autorizado y anonimizado, con separación por periodo o por cliente según la política de datos, evaluación de sesgos y validación por Operación, Riesgos, Cumplimiento y Privacidad.
6. **Licencia.** La licencia del corpus original no debe asumirse automáticamente como licencia de la variante traducida. Verificar las condiciones de reutilización y redistribución de la versión específica.

## Referencias (formato APA 7.ª edición)

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). Efficient intent detection with dual sentence encoders. En *Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI* (pp. 38–45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.nlp4convai-1.5

DeepPavlov. (s. f.). *banking77-translated* [Conjunto de datos; configuración `es`, revisión `48246f6c99b06096d9dbd7fcbaf273c6d82df2f0`]. Hugging Face. https://huggingface.co/datasets/DeepPavlov/banking77-translated/tree/48246f6c99b06096d9dbd7fcbaf273c6d82df2f0

PolyAI. (s. f.). *BANKING77* [Conjunto de datos]. Hugging Face. https://huggingface.co/datasets/PolyAI/banking77
