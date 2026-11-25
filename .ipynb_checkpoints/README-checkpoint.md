# Cell2Cell — Proyecto de Minería de Datos

**Asignatura:** Minería de Datos — Universidad Franz Tamayo
**Dataset:** Cell2Cell (churn de telefonía móvil)
**Metodología:** CRISP-DM

Proyecto que aplica, sobre el dataset de churn de Cell2Cell, todas las técnicas vistas en la cátedra: EDA, regresión logística, árboles de decisión (clasificación y regresión), regresión lineal y clustering con K-Means.

---

## Datos

| Archivo | Filas | Columnas | Uso |
|---|---:|---:|---|
| `cell2celltrain.csv` | 51.047 | 58 | Entrenamiento y evaluación (trae `Churn`) |
| `cell2cellholdout.csv` | 20.000 | 58 | Scoring final (**sin** etiqueta `Churn`) |

**Unidad de análisis:** un cliente (`CustomerID`).
**Target de clasificación:** `Churn` (`No` = 0, `Yes` = 1). Desbalance 71,2% / 28,8%.
**Target de regresión:** `MonthlyRevenue` (ingreso mensual, continuo).

---

## Notebooks del proyecto

| # | Notebook | Tema | Técnica |
|---|---|---|---|
| 1 | `EDA_Cell2Cell.ipynb` | Análisis exploratorio completo | Anatomía, calidad, univariado, bivariado, hipótesis |
| 2 | `Regresion_Logistica_Cell2Cell.ipynb` | Predicción de churn | Regresión logística (statsmodels / sklearn) |
| 3 | `Arboles_de_decision_clasificacion.ipynb` | Predicción de churn | Árbol CART + GridSearchCV + umbral de Youden + ROI |
| 4 | `Regresion_Lineal_Cell2Cell.ipynb` | Explicar el ingreso mensual | OLS + p-valores + supuestos + log |
| 5 | `Arbol_Regresion_Cell2Cell.ipynb` | Predecir el ingreso mensual | Árbol de regresión + poda + curvas de error |
| 6 | `KMeans_Segmentacion_Cell2Cell.ipynb` | Segmentar clientes | K-Means + codo + silhouette + cruce con churn |

### Orden de lectura recomendado

1. **EDA** — primero entender los datos.
2. **Regresión logística** — línea base de clasificación, interpretable.
3. **Árbol de clasificación** — mejor discriminación y lectura de negocio.
4. **Regresión lineal** — el problema continuo: cuánto factura un cliente.
5. **Árbol de regresión** — el mismo problema continuo sin imponer linealidad.
6. **K-Means** — sin etiqueta: qué grupos existen y cuál concentra el riesgo.

---

## Correspondencia con los notebooks de la cátedra

| Notebook de la cátedra | Adaptación en este proyecto |
|---|---|
| `h2_03_entendimiento_datos.ipynb` | `EDA_Cell2Cell.ipynb` (5 pasos del EDA) |
| `h2_04_comprension_negocio_eda.ipynb` | `EDA_Cell2Cell.ipynb` + objetivo de negocio en el notebook de logística |
| `h2_05_EDA.ipynb` | `EDA_Cell2Cell.ipynb` (las 5 preguntas, outliers por IQR, pairplot/heatmap) |
| `h3_01_Regresion_Logistica.ipynb` | `Regresion_Logistica_Cell2Cell.ipynb` |
| `h3_02_Arboles_de_decisión_clasificacion.ipynb` | `Arboles_de_decision_clasificacion.ipynb` |
| `h3_3_Arboles_de_decisión_regresion.ipynb` | `Arbol_Regresion_Cell2Cell.ipynb` |
| `h3_4_Regresion_Lineal_v2.ipynb` | `Regresion_Lineal_Cell2Cell.ipynb` |
| `h4_1_kmeans_ifood.ipynb` | `KMeans_Segmentacion_Cell2Cell.ipynb` |
| `h4_2_kmeans_imagen_restauracion_v2.ipynb` | **No aplica** (ver nota al final) |

---

## Resultados principales

### Clasificación de churn

| Modelo | ROC-AUC | Accuracy | Recall churn |
|---|---:|---:|---:|
| Regresión logística | 0,6088 | 0,5772 | 0,57 |
| Árbol de decisión | 0,6395 | 0,6051 | 0,5925 |

- La métrica de negocio es **ROC-AUC**, porque con 71/29 de desbalance el accuracy engaña.
- El árbol agrega dos cosas que la logística sola no: **umbral óptimo por índice de Youden** y **lectura de margen económico** de la campaña.
- El holdout **no se puede evaluar** (no trae `Churn`): sirve solo para generar `predicciones_holdout.csv`.

### Regresión del ingreso

| Modelo | R² test | RMSE test |
|---|---:|---:|
| Regresión lineal (OLS) | 0,834 | 18,08 USD |
| Árbol de regresión | 0,800 | 20,25 USD |

- El **modelo lineal gana**: el ingreso es una **suma de cargos**, y una suma es lineal por naturaleza.
- El árbol sin podar memoriza (R² = 1,00 en train, 0,73 en test): el notebook muestra el sobreajuste en vivo.
- La transformación **logarítmica no ayudó** (R² 0,73 vs 0,89 en la misma escala). Se probó y se descartó con evidencia.

### Segmentación

| Segmento | Clientes | % base | Churn | Ingreso promedio |
|---|---:|---:|---:|---:|
| Premium alto consumo | 6.001 | 12,0% | 26,8% | 134,8 USD |
| En riesgo (contactó retención) | 1.656 | 3,3% | **45,0%** | 56,2 USD |
| Veterano con equipo viejo | 13.783 | 27,6% | 31,1% | 44,0 USD |
| Base estándar | 28.545 | 57,1% | 27,1% | 49,9 USD |

- K = 4 se eligió combinando codo + silhouette + accionabilidad. El silhouette prefería K = 2 (0,379 vs 0,228), pero dos grupos no permiten diseñar acciones distintas.
- El segmento **"En riesgo"** es chico y concentra el churn: es el target natural de la campaña.
- El segmento **"Veterano con equipo viejo"** confirma desde otra técnica el hallazgo del EDA: equipo viejo → más abandono.

---

## Hallazgos del EDA que sostienen el proyecto

1. **`CurrentEquipmentDays`** es la variable numérica más asociada al churn (r = 0,104) y la relación es monótona: 21,2% de churn con equipos de 0-90 días vs **36,1%** con más de 730 días.
2. **`MadeCallToRetentionTeam = Yes`** → 45,0% de churn vs 28,2% (señal predictiva, no causal).
3. **3.515 celdas faltantes** en 14 variables (ninguna supera el 2%): imputación por mediana/moda.
4. **Sin duplicados**; 9 ingresos negativos y 76 días de equipo negativos (inconsistencias documentadas).
5. **`ServiceArea` con 747 niveles** y **`HandsetPrice` con 56,8% "Unknown"**: variables descartadas.

---

## Cómo ejecutar

Requisitos: Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `jupyter`.

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
jupyter notebook
```

Los notebooks leen los CSV por ruta relativa, así que hay que abrirlos **desde esta carpeta**. Todos están ejecutados con sus salidas visibles: se pueden leer sin correr nada.

---

## Nota sobre `h4_2` (K-Means en imágenes)

El notebook de la cátedra sobre cuantización de imágenes **no tiene una adaptación posible a este dataset**: Cell2Cell es tabular (clientes × variables), no una imagen (píxeles × canales RGB). Forzarlo produciría un ejercicio artificial.

La técnica de fondo —**vector quantization**— sí se aplica acá: la segmentación de clientes agrupa vectores de características en K prototipos, exactamente el mismo principio que agrupa píxeles en K colores. Esa conexión conceptual queda explicada en el marco teórico de `KMeans_Segmentacion_Cell2Cell.ipynb`.

Si la cátedra exige el notebook de imágenes como entregable aparte, se puede reproducir sobre cualquier imagen; no depende de Cell2Cell.
