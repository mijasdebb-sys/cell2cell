# Cell2Cell

**Trabajo práctico de Minería de Datos — Universidad Franz Tamayo**

Este repositorio reúne los notebooks que hice con el dataset Cell2Cell. La idea fue practicar minería de datos con información de clientes de telefonía: explorar los datos, intentar predecir el abandono y el ingreso mensual, y formar grupos de clientes. Usé CRISP-DM como guía.

## Qué contiene

- `cell2celltrain.csv`: 51.047 clientes y 58 columnas. Lo usé para entrenar y evaluar los modelos.
- `cell2cellholdout.csv`: 20.000 clientes y 58 columnas. No trae los valores reales de `Churn`, así que solo sirve para generar predicciones.
- `predicciones_holdout.csv`: predicciones generadas para el conjunto holdout.
- `confusion_matrix.png` y `roc_curve.png`: gráficos que obtuve del modelo de clasificación.

Cada fila representa un cliente (`CustomerID`). En el conjunto de entrenamiento, `Churn` indica si abandonó el servicio (`Yes` o `No`) y el 28,8% de los clientes lo hizo. Para los modelos de ingreso usé `MonthlyRevenue`.

## Notebooks

| Notebook | Qué hice |
|---|---|
| `EDA_Cell2Cell.ipynb` | Exploré los datos, su calidad y las variables relacionadas con el abandono. |
| `Regresion_Logistica_Cell2Cell.ipynb` | Probé la regresión logística para predecir el abandono. |
| `Arboles_de_decision_clasificacion.ipynb` | Probé un árbol de decisión para predecir el abandono y evaluar un escenario simple de campaña. |
| `Regresion_Lineal_Cell2Cell.ipynb` | Usé regresión lineal para intentar explicar y predecir el ingreso mensual. |
| `Arbol_Regresion_Cell2Cell.ipynb` | Comparé un árbol de regresión con la regresión lineal. |
| `KMeans_Segmentacion_Cell2Cell.ipynb` | Agrupé clientes con K-Means y revisé qué grupos tenían más abandono. |

Si estás empezando, sigue el orden de la tabla.

## Algunos resultados

- **Abandono:** en el conjunto de prueba, el árbol de decisión quedó por encima de la regresión logística: ROC-AUC de 0,6395 frente a 0,6088. También detectó algo más de clientes que efectivamente abandonaron: 59,25% frente a 57%.
- **Ingreso mensual:** la regresión lineal quedó mejor, con R² de 0,834 y RMSE de 18,08 USD. El árbol de regresión obtuvo R² de 0,800 y RMSE de 20,25 USD. La versión con transformación logarítmica no dio mejores resultados y la descarté.
- **Segmentos:** K-Means agrupó a los clientes en cuatro segmentos. El grupo “En riesgo” reúne 1.656 clientes y tiene un 45% de abandono, así que podría servir como punto de partida para una campaña de retención.
- **Equipos antiguos:** en el EDA se ve que los clientes con equipos de más de 730 días abandonan más (36,1%) que quienes tienen equipos de 0 a 90 días (21,2%).
- **Retención:** quienes contactaron al equipo de retención muestran un 45% de abandono. Esto no significa que el contacto cause el abandono; es solo una señal que encontré en los datos.

Durante la preparación rellené 3.515 datos faltantes usando mediana o moda. Dejé fuera `ServiceArea` y `HandsetPrice` porque tenían demasiados niveles o muchos valores desconocidos.

## Cómo correrlo

Para ejecutarlo necesitas Python 3.10 o superior con `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn` y `jupyter`.

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
jupyter notebook
```

Abre los notebooks desde esta carpeta para que puedan encontrar los CSV. Todos ya están ejecutados, así que puedes leer los resultados sin correrlos.

## Sobre el notebook de K-Means en imágenes

El ejercicio de K-Means con imágenes no lo adapté a este dataset, porque Cell2Cell contiene datos de clientes y no píxeles. Sí reutilicé la idea general de agrupar vectores para segmentar clientes.
