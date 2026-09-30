import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Cargamos datos
cell2cell_train = pd.read_csv('cell2celltrain.csv', sep=',', encoding='iso-8859-1')
cell2cell_holdout = pd.read_csv('cell2cellholdout.csv', sep=',', encoding='iso-8859-1')

print("Train:", cell2cell_train.shape)
print("Holdout:", cell2cell_holdout.shape)

# Convertimos TARGET
cell2cell_train['Churn'] = cell2cell_train['Churn'].map({'No': 0, 'Yes': 1})
cell2cell_holdout['Churn'] = cell2cell_holdout['Churn'].map({'No': 0, 'Yes': 1})

# Separamos X y
X_train = cell2cell_train.drop(columns=['Churn', 'CustomerID'])
y_train = cell2cell_train['Churn']

X_holdout = cell2cell_holdout.drop(columns=['Churn', 'CustomerID'])

# Tipos
variables_numericas = X_train.select_dtypes(include=np.number).columns
variables_cualitativas = X_train.select_dtypes(exclude=np.number).columns

# Imputación
imputer_num = SimpleImputer(strategy='median')
X_train[variables_numericas] = imputer_num.fit_transform(X_train[variables_numericas])
X_holdout[variables_numericas] = imputer_num.transform(X_holdout[variables_numericas])

imputer_cat = SimpleImputer(strategy='most_frequent')
X_train[variables_cualitativas] = imputer_cat.fit_transform(X_train[variables_cualitativas])
X_holdout[variables_cualitativas] = imputer_cat.transform(X_holdout[variables_cualitativas])

# One-hot encoding
X_train_encoded = pd.get_dummies(X_train, drop_first=True).astype(float)
X_holdout_encoded = pd.get_dummies(X_holdout, drop_first=True).astype(float)

X_train_encoded, X_holdout_encoded = X_train_encoded.align(
    X_holdout_encoded, join='left', axis=1, fill_value=0
)

print("X_train_encoded:", X_train_encoded.shape)
print("X_holdout_encoded:", X_holdout_encoded.shape)
print("Valores faltantes train:", X_train_encoded.isnull().sum().sum())
print("Valores faltantes holdout:", X_holdout_encoded.isnull().sum().sum())

# Pipeline con escalado + class_weight balanced
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
        C=1.0
    ))
])

# Validación interna 80/20
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_encoded, y_train, test_size=0.2, random_state=42, stratify=y_train
)

pipe.fit(X_tr, y_tr)
y_val_pred = pipe.predict(X_val)
y_val_proba = pipe.predict_proba(X_val)[:, 1]

print(f"\n=== VALIDACIÓN INTERNA (80/20) ===")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")
print(f"\nReporte de clasificación:")
print(classification_report(y_val, y_val_pred, target_names=['No Churn', 'Churn']))

# Matriz de confusión
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de Confusión - Validación')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'LogReg (AUC = {roc_auc_score(y_val, y_val_proba):.4f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Validación')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()

# Entrenar modelo final con TODO el train
pipe.fit(X_train_encoded, y_train)
print("\nModelo final entrenado con todo el set de training.")

# Predicciones en holdout (sin etiquetas reales)
y_pred = pipe.predict(X_holdout_encoded)
y_pred_proba = pipe.predict_proba(X_holdout_encoded)[:, 1]

print(f"Distribución predicciones: {pd.Series(y_pred).value_counts().to_dict()}")
print(f"Probabilidad media de churn: {y_pred_proba.mean():.4f}")

# Coeficientes del modelo final
coef_df = pd.DataFrame({
    'Variable': X_train_encoded.columns,
    'Coeficiente': pipe.named_steps['clf'].coef_[0]
})
coef_df['Valor_Absoluto'] = coef_df['Coeficiente'].abs()
top_coef = coef_df.sort_values('Valor_Absoluto', ascending=False).head(15)
print("\nTop 15 coeficientes por valor absoluto (modelo final):")
print(top_coef.to_string(index=False))

# Guardar predicciones
results = pd.DataFrame({
    'CustomerID': cell2cell_holdout['CustomerID'],
    'Churn_Predicho': y_pred,
    'Probabilidad_Churn': y_pred_proba
})
results.to_csv('predicciones_holdout.csv', index=False)
print("\nPredicciones guardadas en 'predicciones_holdout.csv'")
print("Gráficos guardados: confusion_matrix.png, roc_curve.png")
