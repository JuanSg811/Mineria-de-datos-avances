"""

CLASIFICACION CON K VECINOS MAS CERCANOS (KNN) 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\NASA_Exoplanet_Archive_Data.csv" 
OUTPUT_DIR = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Semana_10"
SEMILLA = 42

FEATURES = ["pl_orbper", "pl_bmasse", "st_teff", "st_mass", "sy_dist", "sy_vmag"]
FEATURES_LOG = ["pl_orbper", "pl_bmasse", "sy_dist"] 
CLASES = ["Transit", "Radial Velocity"]


df = pd.read_csv(INPUT_PATH)
df = df[df["default_flag"] == 1]
df = df[df["discoverymethod"].isin(CLASES)]
df = df.dropna(subset=FEATURES)
df = df[(df[FEATURES_LOG] > 0).all(axis=1)]

X = df[FEATURES].copy()
for col in FEATURES_LOG:
    X[col] = np.log10(X[col])
y = df["discoverymethod"]

print("1. DATOS")
print(f"Planetas utilizables (sin datos faltantes en las features): {len(df)}")
print(y.value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMILLA
)
print(f"\nEntrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

# ELECCION DE k MEDIANTE VALIDACION CRUZADA (solo con datos de entrenamiento)

print("\n2. SELECCION DEL MEJOR k (validación cruzada de 5 folds)")

valores_k = list(range(1, 31))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
exactitud_cv = []
for k in valores_k:
    modelo_k = Pipeline([("escala", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=k))])
    puntajes = cross_val_score(modelo_k, X_train, y_train, cv=cv, scoring="accuracy")
    exactitud_cv.append(puntajes.mean())

mejor_k = valores_k[int(np.argmax(exactitud_cv))]
print(f"Mejor k = {mejor_k}  (exactitud CV = {max(exactitud_cv):.4f})")


# MODELO FINAL Y EVALUACION EN CONJUNTO DE PRUEBA

modelo = Pipeline([("escala", StandardScaler()),
                   ("knn", KNeighborsClassifier(n_neighbors=mejor_k))])
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)


print(f"\n3. EVALUACION EN CONJUNTO DE PRUEBA (k = {mejor_k})")
print(f"Exactitud (accuracy): {accuracy_score(y_test, y_pred):.4f}")

# Línea base: ¿qué exactitud obtendría un modelo que siempre predice la clase mayoritaria?
base = y_test.value_counts(normalize=True).max()
print(f"Línea base (siempre predecir '{y_test.value_counts().idxmax()}'): {base:.4f}")

print("\nReporte de clasificación (precisión, recall, F1 por clase):")
print(classification_report(y_test, y_pred, digits=3))

matriz = confusion_matrix(y_test, y_pred, labels=CLASES)
print("Matriz de confusión (filas = real, columnas = predicho):")
print(pd.DataFrame(matriz, index=[f"real {c}" for c in CLASES],
                   columns=[f"pred {c}" for c in CLASES]))

#Graficas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(valores_k, exactitud_cv, marker="o", color="#2c6e91")
axes[0].axvline(mejor_k, color="#d1495b", linestyle="--", label=f"Mejor k = {mejor_k}")
axes[0].set_title("Exactitud (validación cruzada) según k")
axes[0].set_xlabel("Número de vecinos (k)")
axes[0].set_ylabel("Exactitud")
axes[0].grid(alpha=0.3)
axes[0].legend()

ConfusionMatrixDisplay(matriz, display_labels=CLASES).plot(ax=axes[1], cmap="Blues",
                                                            colorbar=False)
axes[1].set_title(f"Matriz de confusión (conjunto de prueba, k={mejor_k})")

fig.tight_layout()
ruta_salida = f"{OUTPUT_DIR}/clasificacion_knn.png"
fig.savefig(ruta_salida, dpi=110)
plt.close(fig)
print(f"\nGráfica guardada: {ruta_salida}")
