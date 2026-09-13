
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Dataset_Limpiado.csv" 
OUTPUT_DIR = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos"

plt.rcParams["figure.dpi"] = 110
COLOR_PRINCIPAL = "#2c6e91"

df = pd.read_csv(INPUT_PATH)
print(f"Dataset cargado: {df.shape[0]} planetas x {df.shape[1]} columnas")


metodos_principales = (
    df["discoverymethod"].value_counts().head(5).index.tolist()
)
print("Métodos principales usados en las gráficas:", metodos_principales)

# Descubrimientos por año
def grafica_lineas_por_anio(df: pd.DataFrame, ax):
    conteo_anual = df.groupby("disc_year").size()
    ax.plot(conteo_anual.index, conteo_anual.values, color=COLOR_PRINCIPAL, marker="o", markersize=3)
    ax.set_title("1. Número de descubrimientos por año")
    ax.set_xlabel("Año de descubrimiento")
    ax.set_ylabel("Cantidad de exoplanetas")
    ax.grid(alpha=0.3)

# Métodos de descubrimiento más utilizados

def grafica_barras_metodos(df: pd.DataFrame, ax):
    conteo_metodo = df["discoverymethod"].value_counts()
    ax.bar(conteo_metodo.index, conteo_metodo.values, color=COLOR_PRINCIPAL)
    ax.set_title("2. Métodos de descubrimiento más utilizados")
    ax.set_xlabel("Método de descubrimiento")
    ax.set_ylabel("Cantidad de exoplanetas")
    ax.tick_params(axis="x", rotation=75)
    ax.set_yscale("log")  

# Evolución de cada método a través del tiempo

def grafica_evolucion_metodos(df: pd.DataFrame, ax):
    subconjunto = df[df["discoverymethod"].isin(metodos_principales)]
    tabla_pivote = (
        subconjunto.groupby(["disc_year", "discoverymethod"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=metodos_principales)
    )
    ax.stackplot(tabla_pivote.index, tabla_pivote.T.values, labels=tabla_pivote.columns, alpha=0.85)
    ax.set_title("3. Evolución de cada método a través del tiempo (área apilada)")
    ax.set_xlabel("Año de descubrimiento")
    ax.set_ylabel("Cantidad de exoplanetas")
    ax.legend(loc="upper left", fontsize=8)

# Radio del planeta según el método

def grafica_caja_radio(df: pd.DataFrame, ax):
    subconjunto = df[df["discoverymethod"].isin(metodos_principales)]
    datos_por_metodo = [
        subconjunto.loc[subconjunto["discoverymethod"] == m, "pl_rade"].dropna()
        for m in metodos_principales
    ]
    ax.boxplot(datos_por_metodo, tick_labels=metodos_principales, showfliers=True)
    ax.set_title("4. Radio del planeta según el método (pl_rade)")
    ax.set_xlabel("Método de descubrimiento")
    ax.set_ylabel("Radio (radios terrestres)")
    ax.set_yscale("log")  # log porque hay planetas rocosos y gigantes en la misma escala
    ax.tick_params(axis="x", rotation=75)

# Masa vs. radio

def grafica_dispersion_masa_radio(df: pd.DataFrame, ax):
    subconjunto = df[df["discoverymethod"].isin(metodos_principales)]
    for metodo in metodos_principales:
        datos = subconjunto[subconjunto["discoverymethod"] == metodo]
        ax.scatter(datos["pl_rade"], datos["pl_bmasse"], label=metodo, alpha=0.5, s=15)
    ax.set_title("5. Masa vs. radio del planeta")
    ax.set_xlabel("Radio (radios terrestres)")
    ax.set_ylabel("Masa (masas terrestres)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

graficas = [
    ("01_lineas_descubrimientos_por_año", grafica_lineas_por_anio),
    ("02_barras_metodos", grafica_barras_metodos),
    ("03_evolucion_metodos_tiempo", grafica_evolucion_metodos),
    ("04_caja_radio_por_metodo", grafica_caja_radio),
    ("05_dispersion_masa_radio", grafica_dispersion_masa_radio),
]

for nombre_archivo, funcion_grafica in graficas:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    funcion_grafica(df, ax)
    fig.tight_layout()
    ruta_salida = f"{OUTPUT_DIR}/{nombre_archivo}.png"
    fig.savefig(ruta_salida)
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_salida}")

print("\nGráficas generadas.")
