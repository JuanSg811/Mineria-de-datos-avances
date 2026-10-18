
# AGRUPAMIENTO CON K-MEANS - NASA Exoplanet Archive


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

pd.set_option("display.width", 140)

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\NASA_Exoplanet_Archive_Data.csv" 
OUTPUT_DIR = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Semana_12"
SEMILLA = 42
K_ELEGIDO = 3  

FEATURES = ["pl_rade", "pl_bmasse", "pl_orbper", "st_teff"]
FEATURES_LOG = ["pl_rade", "pl_bmasse", "pl_orbper"]


# 1. PREPARACION DE DATOS

df = pd.read_csv(INPUT_PATH)
df = df[df["default_flag"] == 1]
df = df.dropna(subset=FEATURES)
df = df[(df[FEATURES_LOG] > 0).all(axis=1)].copy()

X = df[FEATURES].copy()
for col in FEATURES_LOG:
    X[col] = np.log10(X[col])
X_esc = StandardScaler().fit_transform(X)

print("\n1. DATOS")
print(f"Planetas utilizables (sin datos faltantes): {len(df)}")


# 2. ELECCION DE k: CODO + SILUETA

print("\n2. SELECCION DE k (método del codo y coeficiente de silueta)")

rango_k = list(range(2, 11))
inercias, siluetas = [], []
for k in rango_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X_esc)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_esc, km.labels_))

print(f"{'k':>3} {'inercia':>10} {'silueta':>9}")
for k, i, s in zip(rango_k, inercias, siluetas):
    print(f"{k:>3} {i:>10.1f} {s:>9.3f}")

k_max_silueta = rango_k[int(np.argmax(siluetas))]
print(f"\nMayor silueta: k = {k_max_silueta} ({max(siluetas):.3f}).")
print(f"Se elige k = {K_ELEGIDO}: su silueta es casi igual a la máxima, el codo "
      f"se aprecia alrededor de k = 3-4, y 3 grupos permiten una interpretación "
      f"física más rica que solo 2.")


# 3. MODELO FINAL

modelo = KMeans(n_clusters=K_ELEGIDO, n_init=10, random_state=SEMILLA).fit(X_esc)
df["cluster"] = modelo.labels_
silueta_final = silhouette_score(X_esc, modelo.labels_)

print(f"\n3. MODELO FINAL (k = {K_ELEGIDO})")
print(f"Coeficiente de silueta: {silueta_final:.3f}")
print(f"Inercia: {modelo.inertia_:.1f}")
print("\nTamaño de cada cluster:")
print(df["cluster"].value_counts().sort_index().to_string())


# 4. PERFIL DE CADA CLUSTER 

print("\n 4. PERFIL DE CADA CLUSTER (mediana en unidades originales)")
perfil = df.groupby("cluster")[FEATURES].median().round(2)
perfil["n_planetas"] = df.groupby("cluster").size()
print(perfil.to_string())


# 5. COMPARACION CON LA CATEGORIA POR RADIO

df["pl_categoria"] = np.select(
    [df["pl_rade"] < 1.5, df["pl_rade"] < 2.0, df["pl_rade"] < 6.0],
    ["Rocoso", "Super-Tierra", "Mini-Neptuno"],
    default="Gigante",
)

print("\n5. CLUSTERS vs. CATEGORIA POR RADIO")
print(pd.crosstab(df["cluster"], df["pl_categoria"]).to_string())
ari = adjusted_rand_score(df["pl_categoria"], df["cluster"])
print(f"\nÍndice de Rand Ajustado (ARI): {ari:.3f}  "
      f"(1 = coinciden perfectamente, 0 = acuerdo aleatorio)")


# 6. GRAFICAS

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].plot(rango_k, inercias, marker="o", color="#2c6e91")
axes[0].axvline(K_ELEGIDO, color="#d1495b", linestyle="--", label=f"k elegido = {K_ELEGIDO}")
axes[0].set_title("Método del codo")
axes[0].set_xlabel("Número de clusters (k)")
axes[0].set_ylabel("Inercia")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(rango_k, siluetas, marker="o", color="#2c6e91")
axes[1].axvline(K_ELEGIDO, color="#d1495b", linestyle="--", label=f"k elegido = {K_ELEGIDO}")
axes[1].set_title("Coeficiente de silueta")
axes[1].set_xlabel("Número de clusters (k)")
axes[1].set_ylabel("Silueta promedio")
axes[1].legend()
axes[1].grid(alpha=0.3)

pca = PCA(n_components=2, random_state=SEMILLA)
componentes = pca.fit_transform(X_esc)
dispersion = axes[2].scatter(componentes[:, 0], componentes[:, 1], c=df["cluster"],
                              cmap="viridis", alpha=0.6, s=18)
centros_pca = pca.transform(modelo.cluster_centers_)
axes[2].scatter(centros_pca[:, 0], centros_pca[:, 1], c="red", marker="X", s=200,
                edgecolor="black", label="Centroides")
axes[2].set_title(f"Clusters en 2D (PCA, {pca.explained_variance_ratio_.sum()*100:.0f}% "
                  f"de varianza)")
axes[2].set_xlabel("Componente principal 1")
axes[2].set_ylabel("Componente principal 2")
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.tight_layout()
ruta_salida = f"{OUTPUT_DIR}/clustering_kmeans.png"
fig.savefig(ruta_salida, dpi=110)
plt.close(fig)
print(f"\nGráfica guardada: {ruta_salida}")
