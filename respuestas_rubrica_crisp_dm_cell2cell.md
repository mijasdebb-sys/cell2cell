# Respuestas de la rúbrica CRISP-DM: Cell2Cell

Todas las respuestas están justificadas con referencias explícitas al código y a las salidas visibles de los dos notebooks del proyecto:

- **NB-Tree** = `Arboles_de_decision_clasificacion.ipynb`
- **NB-Log** = `Regresion_Logistica_Cell2Cell.ipynb`

Cada referencia indica la celda por su `id` interno y su número de ejecución (`In [n]`), de modo que cualquiera pueda abrir el notebook y verificar el código y el resultado.

---

## 1. Entendimiento del negocio

### 1.1 Objetivo de negocio

**Respuesta:** predecir si un cliente abandonará el servicio (`Churn`) para priorizar una campaña de retención y decidir a qué clientes contactar primero.

> **Evidencia — NB-Tree, celda `intro` (markdown de título):**
>
> *"**Objetivo:** predecir `Churn` (abandono del cliente) con un `DecisionTreeClassifier`, ajustar hiperparámetros con `GridSearchCV`, calcular el umbral óptimo vía ROC y evaluar el margen de negocio de la campaña de retención."*

La decisión de negocio que se apoya es la asignación del presupuesto de contacto de la campaña de retención, tal como lo formaliza después el escenario económico del propio NB-Tree (sección 1.4 de este documento y celda `biz_params`, `In [10]`).

### 1.2 Métrica principal

**Respuesta:** la métrica principal del modelado es **ROC-AUC**.

- **Qué mide:** la capacidad del modelo para ordenar a los churners (`Churn=1`) por encima de los no churners (`Churn=0`), en todos los umbrales posibles.
- **Por qué se eligió:** `Churn` está desbalanceado (71,18% vs 28,82%, ver EDA-4), por lo que accuracy puro engaña; AUC es invariante al umbral y apta para clases desbalanceadas.
- **Unidad:** adimensional, escala 0–1.

> **Evidencia — NB-Tree, celda `grid` (`In [5]`), extracto del código:**
>
> ```python
> grid = GridSearchCV(
>     estimator=DecisionTreeClassifier(random_state=42),
>     param_grid=param_grid,
>     scoring='roc_auc',   # <== métrica de selección
>     cv=3,
>     n_jobs=-1
> )
> ```
>
> **Salida visible:** `Mejor AUC-ROC (CV): 0.6365`

> **Evidencia — NB-Log, celda `metrics` (`In [11]`), extracto del código:**
>
> ```python
> print('ROC-AUC: ', round(roc_auc_score(y_val, y_val_proba), 4))
> ```
>
> **Salida visible:** `ROC-AUC:  0.6088`

Además, el mismo NB-Tree justifica textualmente la elección en la celda markdown `grid_md`:

> *"Probamos combinaciones de `max_depth`, `min_samples_leaf` y `criterion` usando `roc_auc` como métrica de selección (apta para clases desbalanceadas)."*

### 1.3 Definición y cálculo de la métrica

**Fórmula/procedimiento:** ROC-AUC es el área bajo la curva que relaciona `TPR` (recall de churners) contra `FPR`, calculada con las probabilidades del modelo.

Variables que intervienen:

1. `y_test` / `y_val`: etiqueta real binaria (`Churn` mapeado `No=0`, `Yes=1` — NB-Tree celda `load`, `In [12]`; NB-Log celda `map_target`, `In [4]`).
2. `y_proba` / `y_val_proba`: probabilidad predicha de `Churn=1`.

> **Evidencia — NB-Tree, celda `proba` (`In [8]`), extracto del código:**
>
> ```python
> y_proba = best_tree.predict_proba(X_test)[:, 1]
>
> # Curva ROC y umbral óptimo (índice de Youden: tpr - fpr)
> fpr, tpr, thresholds = roc_curve(y_test, y_proba)
> youden = tpr - fpr
> optimal_threshold = thresholds[np.argmax(youden)]
>
> print(f'ROC-AUC test: {roc_auc_score(y_test, y_proba):.4f}')
> print(f'Umbral óptimo (Youden): {optimal_threshold:.4f}')
> ```
>
> **Salida visible:**
>
> ```text
> ROC-AUC test: 0.6395
> Umbral óptimo (Youden): 0.5233
> ```

**Valor obtenido para el conjunto de datos** (unificando ambos notebooks):

| Modelo | ROC-AUC | Dónde se calcula |
|---|---:|---|
| Regresión logística | 0,6088 | NB-Log, celda `metrics` (`In [11]`), validación interna 80/20 |
| Árbol de decisión | 0,6365 | NB-Tree, celda `grid` (`In [5]`), mejor score de CV |
| Árbol de decisión | 0,6395 | NB-Tree, celda `proba` (`In [8]`), test interno |

### 1.4 Mejora tentativa (opcional)

Como **meta futura**, no como resultado observado, se propone alcanzar `ROC-AUC >= 0.70` en una evaluación independiente, manteniendo o mejorando el recall actual de `0,5925`. Es razonable como objetivo porque la grilla actual (NB-Tree, celda `grid`, `In [5]`) es acotada (`max_depth` hasta 10, solo un árbol); ampliar la búsqueda o usar ensambles permitiría ganar discriminación sin rehacer la preparación de datos.

> Esta propuesta se identifica como escenario futuro, siguiendo la indicación de la rúbrica de no presentarla como resultado observado.

---

## 2. Entendimiento de los datos y EDA

### EDA-1. ¿Qué datos tengo?

**Respuesta:** dos archivos CSV locales con 58 variables cada uno:

- `cell2celltrain.csv`: 51.047 clientes con etiqueta real `Churn`.
- `cell2cellholdout.csv`: 20.000 clientes sin etiqueta (scoring final).

La unidad de análisis es un cliente identificado por `CustomerID`.

> **Evidencia — NB-Tree, celda `load` (`In [12]`), extracto del código:**
>
> ```python
> cell2cell_train = pd.read_csv('cell2celltrain.csv', sep=',', encoding='iso-8859-1')
> cell2cell_holdout = pd.read_csv('cell2cellholdout.csv', sep=',', encoding='iso-8859-1')
>
> print('Train:', cell2cell_train.shape)
> print('Holdout:', cell2cell_holdout.shape)
>
> # Convertimos TARGET a 0/1
> cell2cell_train['Churn'] = cell2cell_train['Churn'].map({'No': 0, 'Yes': 1})
>
> print('\nDistribución Churn:')
> print(cell2cell_train['Churn'].value_counts(normalize=True) * 100)
> ```
>
> **Salida visible:**
>
> ```text
> Train: (51047, 58)
> Holdout: (20000, 58)
>
> Distribución Churn:
> Churn
> 0    71.18146
> 1    28.81854
> ```

> **Evidencia — NB-Log, celda `types` (`In [7]`), código que tipifica las variables:**
>
> ```python
> variables_numericas = X_train.select_dtypes(include=np.number).columns
> variables_cualitativas = X_train.select_dtypes(exclude=np.number).columns
> ```
>
> **Salida visible:** `Variables numéricas (34)` y `Variables cualitativas (22)`, con las listas completas impresas.

**Tipos principales:** 34 variables numéricas (ingresos, minutos, llamadas, antigüedad) y 22 categóricas (demografía, equipo, retención). Después de excluir `Churn` y `CustomerID` quedan 56 predictoras, y tras one-hot encoding, 829 columnas.

> **Evidencia — NB-Tree, celda `prep` (`In [3]`), extracto del código:**
>
> ```python
> X = cell2cell_train.drop(columns=['Churn', 'CustomerID'])
> ...
> X_encoded = pd.get_dummies(X, drop_first=True).astype(float)
> print('Shape final X:', X_encoded.shape)
> ```
>
> **Salida visible:** `Shape final X: (51047, 829)` — `Nulos restantes: 0`

**Profundidad adicional (derivación aritmética de las salidas citadas):**

- De los 71.047 clientes totales del proyecto, el train representa el 71,85% (51.047) y el holdout el 28,15% (20.000).
- El salto de 56 a 829 columnas lo produce el one-hot encoding de las 22 categóricas; `ServiceArea` es la que más niveles aporta (la tabla de coeficientes, NB-Log `In [17]`, muestra decenas de filas `ServiceArea_*`).
- Ambos archivos comparten las mismas 58 columnas, lo que permite aplicar el mismo pipeline de imputación/codificación a los dos (NB-Log `impute`/`encode`, `In [8]`–`In [9]`).

### EDA-2. ¿Hay datos faltantes?

**Respuesta:** sí. En el train hay 3.515 valores faltantes repartidos en 14 variables; en el holdout la columna `Churn` está vacía por diseño (scoring) y también hay faltantes en predictoras. El tratamiento aplicado fue imputación por **mediana** (numéricas) y por **moda** (categóricas).

> **Evidencia — NB-Log, celda `missing` (`In [5]`), código y salida completa:**
>
> ```python
> # Revisamos valores faltantes
> missing = cell2cell_train.isnull().sum()
> missing[missing > 0].sort_values(ascending=False)
> ```
>
> **Salida visible (tabla):**
>
> ```text
> AgeHH2                   909
> AgeHH1                   909
> PercChangeRevenues       367
> PercChangeMinutes        367
> MonthlyRevenue           156
> MonthlyMinutes           156
> RoamingCalls             156
> OverageMinutes           156
> DirectorAssistedCalls    156
> TotalRecurringCharge     156
> ServiceArea               24
> Handsets                   1
> CurrentEquipmentDays       1
> HandsetModels              1
> ```

> **Evidencia — NB-Log, celda `impute` (`In [8]`), código completo del tratamiento:**
>
> ```python
> # Imputación de valores faltantes
> imputer_num = SimpleImputer(strategy='median')
> X_train[variables_numericas] = imputer_num.fit_transform(X_train[variables_numericas])
> X_holdout[variables_numericas] = imputer_num.transform(X_holdout[variables_numericas])
>
> imputer_cat = SimpleImputer(strategy='most_frequent')
> X_train[variables_cualitativas] = imputer_cat.fit_transform(X_train[variables_cualitativas])
> X_holdout[variables_cualitativas] = imputer_cat.transform(X_holdout[variables_cualitativas])
> ```

> **Evidencia — NB-Log, celda `encode` (`In [9]`), verificación posterior:**
>
> ```python
> print('Valores faltantes train:', X_train_encoded.isnull().sum().sum())
> print('Valores faltantes holdout:', X_holdout_encoded.isnull().sum().sum())
> ```
>
> **Salida visible:** `Valores faltantes train: 0` — `Valores faltantes holdout: 0`

**Impacto sobre la métrica:** al imputar sobre todo el train antes del split interno (NB-Tree, celda `prep`, `In [3]`) existe un leve riesgo de fuga de información; aun así, los resultados son coherentes entre validación cruzada (0,6365, `In [5]`) y test (0,6395, `In [8]`), lo que sugiere que el efecto es pequeño.

**Profundidad adicional (derivación aritmética de la tabla citada):**

- `AgeHH1`/`AgeHH2`: 909 faltantes cada una = 1,78% de 51.047 — la mayor incidencia, aunque aún baja.
- `PercChangeRevenues`/`PercChangeMinutes`: 367 cada una = 0,72%.
- Seis variables de facturación/llamadas (`MonthlyRevenue`, `MonthlyMinutes`, `RoamingCalls`, `OverageMinutes`, `DirectorAssistedCalls`, `TotalRecurringCharge`): 156 cada una = 0,31%.
- `ServiceArea`: 24 = 0,05%. `Handsets`, `HandsetModels`, `CurrentEquipmentDays`: 1 cada una ≈ 0,002%.
- En total son 3.515 celdas faltantes ≈ 0,12% de las 2.960.726 celdas del train (51.047 × 58).
- Lectura para la diapositiva: los faltantes son pocos y concentrados en pocas variables. Imputar con mediana/moda en vez de eliminar filas preserva los 51.047 registros, lo que importa porque la clase minoritaria (`Churn`) ya es escasa (14.711 casos).

### EDA-3. ¿Existen valores atípicos, duplicados o inconsistencias?

**Respuesta honesta:** con el código actualmente ejecutado **no hay una comprobación explícita** de duplicados (`duplicated()`), atípicos (IQR/boxplot/z-score) ni reglas de inconsistencia. Por eso, la respuesta justificada para la presentación es:

> *"Los notebooks actuales no incluyen una revisión explícita de atípicos, duplicados o inconsistencias; por lo tanto, no podemos afirmar si existen o no sin ejecutar esa revisión. Lo que sí está justificado con código es que: (a) los faltantes fueron imputados y verificados en cero (NB-Log, celda `encode`, `In [9]`); (b) `CustomerID` se excluyó del modelado por ser identificador (NB-Tree, celda `prep`, `In [3]`); (c) las categorías se codificaron con `pd.get_dummies` sin reportar errores (celdas `prep`/`encode`). El paso pendiente es agregar `cell2cell_train.duplicated().sum()` y un análisis IQR sobre las columnas numéricas para completar esta pregunta con evidencia."*

Los tres puntos citados están respaldados por las celdas referenciadas en EDA-1 y EDA-2.

### EDA-4. ¿Cómo se distribuyen los datos?

**Respuesta:** la distribución documentada con gráfico y valores es la del **target `Churn`**, que muestra un desbalance: 71,18% `No` (36.336 clientes) contra 28,82% `Yes` (14.711 clientes).

> **Evidencia — NB-Log, celda `target` (`In [3]`), código y salida:**
>
> ```python
> # Distribución del TARGET
> print(cell2cell_train['Churn'].value_counts())
> print('\nPorcentaje:')
> print(cell2cell_train['Churn'].value_counts(normalize=True) * 100)
>
> sns.countplot(data=cell2cell_train, x='Churn')
> plt.title('Distribución de Churn')
> plt.xlabel('Churn')
> plt.ylabel('Cantidad de clientes')
> plt.show()
> ```
>
> **Salida visible:**
>
> ```text
> Churn
> No     36336
> Yes    14711
>
> Porcentaje:
> Churn
> No     71.18146
> Yes    28.81854
> ```
>
> Más el gráfico de barras generado por `sns.countplot`.

**Consecuencia documentada en el código:** este desbalance motivó el uso de `class_weight='balanced'` en ambos modelos:

- NB-Log, celda `train` (`In [10]`): `LogisticRegression(..., class_weight='balanced', ...)`.
- NB-Tree, celda `grid` (`In [5]`): `'class_weight': [None, 'balanced']` en el `param_grid`; la mejor combinación elegida incluye `'class_weight': 'balanced'` (salida visible de `grid.best_params_`).

**Limitación a declarar oralmente:** los notebooks no incluyen histogramas o gráficos de densidad de las variables numéricas, por lo que no se puede describir la forma de esas distribuciones sin agregar código nuevo.

**Profundidad adicional para la diapositiva (todo derivado de salidas citadas):**

- **El desbalance define la lectura correcta de las métricas.** Un modelo ingenuo que dijera "no abandona" a todos tendría 71,18% de accuracy (salida de `target`, `In [3]`) y detectaría 0% de churners. La regresión logística balanceada tiene accuracy 0,5772 —*menor* que ese baseline— pero recupera ~57% de los churners; el propio notebook lo advierte en su celda markdown `interp_md`: sin `class_weight='balanced'` el recall de churn colapsaba a ~3%.
- **La partición interna es idéntica para los dos modelos**, lo que hace justa la comparación: ambos reportes (NB-Log `In [11]` y NB-Tree `In [9]`) muestran `support` idéntico (7.268 No Churn / 2.942 Churn sobre 10.210 casos) porque ambos usan `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
- **Cómo queda el holdout (scoring real):** el modelo final predice 10.918 clientes como No Churn y 9.082 como Churn (45,4% del holdout), con probabilidad media de churn 0,4845 (NB-Log `predict_holdout`, `In [15]`). Que ese 45,4% supere al 28,8% histórico es efecto esperado del balanceo: el modelo sobre-alerta a cambio de no perder churners. Para la campaña, esto implica que el umbral de decisión importa tanto como el modelo mismo.

### EDA-5. ¿Qué relaciones existen entre variables?

**Respuesta (predictiva, no causal):** las relaciones observables en las salidas existentes son las que cada modelo identifica como relevantes para `Churn`.

Relación 1 — desde el árbol:

> **Evidencia — NB-Tree, celda `feat_imp` (`In [6]`), código y salida (Top 15):**
>
> ```python
> importancias = pd.DataFrame({
>     'Variable': X_train.columns,
>     'Importancia': best_tree.feature_importances_
> }).sort_values('Importancia', ascending=False)
> display(importancias.head(15))
> ```
>
> **Salida visible (extracto):**
>
> ```text
> CurrentEquipmentDays     0.312608
> MonthsInService          0.183153
> PercChangeMinutes        0.097683
> MonthlyMinutes           0.076394
> TotalRecurringCharge     0.047407
> PercChangeRevenues       0.040119
> ```

Relación 2 — desde la regresión logística:

> **Evidencia — NB-Log, celda `coefs` (`In [17]`), código y salida (Top 15):**
>
> ```python
> coeficientes = pd.DataFrame({
>     'Variable': X_train_encoded.columns,
>     'Coeficiente': pipe.named_steps['clf'].coef_[0]
> })
> coeficientes['Valor_Absoluto'] = coeficientes['Coeficiente'].abs()
> coeficientes.sort_values('Valor_Absoluto', ascending=False).head(15)
> ```
>
> **Salida visible (extracto):**
>
> ```text
> CurrentEquipmentDays   0.387725
> UniqueSubs             0.237290
> MonthsInService       -0.218395
> PercChangeMinutes     -0.151168
> MonthlyMinutes        -0.133885
> ActiveSubs            -0.124388
> ```

**Qué se observa:** en la regresión logística, más antigüedad del servicio (`MonthsInService` negativo) y más uso (`MonthlyMinutes` negativo) se asocian a menor probabilidad de churn, mientras que más días con el equipo actual (`CurrentEquipmentDays` positivo) se asocian a mayor probabilidad. El propio NB-Log lo resume en su celda markdown `interp_md`:

> *"**Top predictores negativos de churn:** `MonthsInService`, `PercChangeMinutes`, `MonthlyMinutes`, `ActiveSubs` — clientes más antiguos y con mayor uso tienden a no abandonar."*

**Limitaciones:** ninguna de estas dos evidencias prueba causalidad. La importancia del árbol no indica dirección (positiva o negativa), y los coeficientes son condicionales al resto del modelo y a la codificación one-hot (advertencia del propio NB-Log en `interp_md` para los niveles de `ServiceArea_*`).

**Profundidad adicional (derivaciones aritméticas de salidas citadas):**

- **Concentración de la señal:** las 6 primeras variables del árbol acumulan 0,757 de la importancia total (0,3126 + 0,1832 + 0,0977 + 0,0764 + 0,0474 + 0,0401); `CurrentEquipmentDays` sola explica el 31,3%. La señal de churn existe en muchas variables, pero está dominada por la antigüedad del equipo y del servicio.
- **La señal está regularizada, no sobreajustada:** el árbol ganador usa `max_depth=10` y `min_samples_leaf=200` (salida de `grid`, `In [5]`): ninguna hoja se construye con menos de 200 clientes, lo que evita reglas hiperespecíficas. Existe además la visualización de los primeros 3 niveles (celda `tree_plot`, `In [7]`), utilizable como figura de la diapositiva.
- **Lectura de negocio de la matriz de confusión** (NB-Tree `In [9]` + `In [10]`): de los 4.576 contactados, 2.833 son falsos positivos → 61,9% del esfuerzo de contacto va a clientes que no iban a abandonar. Con los supuestos del notebook (ingreso por churner retenido $581,52 vs costo por contacto $30), el punto de equilibrio exige recuperar ingresos equivalentes a solo $78,76 por churner contactado ($137.280 / 1.743 VP), un margen de seguridad de ~7,4× sobre el costo supuesto. Esta aritmética explica *por qué* un precision "bajo" (0,38) sigue siendo rentable: el churner recuperado vale mucho más que el contacto desperdiciado.
- **Trade-off casi igual en ambos modelos:** árbol `precision churn 0,38 / recall 0,59` vs regresión `0,36 / 0,57` (reportes `In [9]` e `In [11]`): la diferencia entre modelos en la clase que le importa al negocio es pequeña; lo que más mueve la aguja es el umbral de decisión, y solo el árbol lo optimiza (Youden 0,5233, `In [8]`).

### EDA-6. Hallazgo interesante

**Hallazgo:** `CurrentEquipmentDays` (antigüedad del equipo actual) es la variable dominante en **dos modelos independientes**, lo que le da robustez:

1. Mayor **importancia** del árbol: `0,312608` (NB-Tree, celda `feat_imp`, `In [6]`).
2. **Coeficiente positivo** de mayor magnitud en la regresión logística: `0,387725` (NB-Log, celda `coefs`, `In [17]`).

**Por qué es útil para el negocio:** sugiere priorizar ofertas de renovación de equipo para clientes con equipos antiguos, coherente con la campaña de retención del objetivo (sección 1.1).

**Postura crítica a declarar oralmente:** es una relación **predictiva**, no una prueba de causa. Antes de convertirla en política de negocio conviene validarla con un piloto de contacto o un análisis segmentado por antigüedad del servicio.

**Robustez del hallazgo:** dos algoritmos con inductivas distintas (cortes por umbral vs. frontera lineal) coinciden en colocar `CurrentEquipmentDays` en primer lugar; ambos usan la misma base imputada y codificada a 829 columnas (NB-Tree `prep`, `In [3]`; NB-Log `encode`, `In [9]`), aunque el árbol se ajusta sobre el 80% interno y la regresión final se reentrena con el 100% (NB-Log `final_train`, `In [14]`). La coincidencia, entonces, no se explica por datos diferentes.

**Guion sugerido para la diapositiva 2 (~40 segundos, todo con respaldo en celdas citadas):**

1. "Tenemos 71.047 clientes y el 28,8% abandona" (NB-Log `target`, `In [3]`).
2. "Solo faltan 3.515 celdas (0,12%); imputamos y verificamos cero" (NB-Log `In [5]` y `In [9]`).
3. "Aún no comprobamos atípicos ni duplicados: es nuestra limitación declarada" (sección EDA-3).
4. "Dos modelos independientes coinciden: la antigüedad del equipo es la señal #1" (NB-Tree `In [6]`; NB-Log `In [17]`).
5. Cierre con el gráfico de barras de `Churn` y, si entra, el árbol de 3 niveles (`In [7]`) como imagen.

---

## 3. Conclusión comparativa

Con la evidencia citada celda por celda:

| Aspecto | Regresión logística | Árbol de decisión |
|---|---|---|
| ROC-AUC | 0,6088 (NB-Log, `In [11]`) | 0,6395 (NB-Tree, `In [8]`) |
| Accuracy | 0,5772 (NB-Log, `In [11]`) | 0,6051 (NB-Tree, `In [9]`) |
| Recall churn | 0,57 (reporte de `In [11]`) | 0,5925 (NB-Tree, `In [9]`) |
| Interpretación | coeficientes (`coefs`, `In [17]`) | importancias (`feat_imp`, `In [6]`) |
| Umbral | fijo (`predict`, `In [11]`) | óptimo por Youden (`In [8]`) |
| Evaluación económica | no incluye | margen $876.309 (`biz_params`, `In [10]`) |

**Conclusión:** el árbol tiene mejor discriminación (0,6395 vs 0,6088 sobre la misma partición interna 80/20 estratificada, `random_state=42`) y agrega dos elementos que la regresión logística sola no aporta: umbral optimizado por Youden y lectura de margen de negocio. La regresión logística, en cambio, es la que permite interpretar la dirección de cada relación.

**Detalle a declarar siempre:** el holdout (`cell2cellholdout.csv`) se usa solo para generar predicciones (`predicciones_holdout.csv`, NB-Log, celda `save_pred`, `In [16]`) porque no contiene la etiqueta real de `Churn`; ahí no hay accuracy ni AUC calculables, y así lo anticipa el markdown de título del propio NB-Log.
