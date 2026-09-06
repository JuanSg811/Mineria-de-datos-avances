
import pandas as pd
import numpy as np


INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Dataset_Limpiado.csv"

df = pd.read_csv(INPUT_PATH)

print(f"Dataset cargado: {df.shape[0]} registros y {df.shape[1]} columnas\n")

def obtener_moda(serie):
    """
    Obtiene la primera moda de una serie.
    """

    serie = serie.dropna()
    moda = serie.mode()

    if not moda.empty:
        return moda.iloc[0]

    return np.nan


def obtener_kurtosis(serie):
    """
    Calcula la kurtosis de una serie.
    """

    return serie.kurt()

# Características físicas, orbitales y estelares generales de los exoplanetas registrados

# Variables orbitales
variables_orbitales = [
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen"
]


# Variables físicas del planeta
variables_fisicas = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_insol"
]


# Variables estelares
variables_estelares = [
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met"
]


# Uno todas las variables
variables_pregunta_1 = (
    variables_orbitales
    + variables_fisicas
    + variables_estelares
)


# Tabla de estadísticas descriptivas
estadisticas_generales = pd.DataFrame({

    "Conteo":
        df[variables_pregunta_1].count(),

    "Mínimo":
        df[variables_pregunta_1].min(),

    "Máximo":
        df[variables_pregunta_1].max(),

    "Media":
        df[variables_pregunta_1].mean(),

    "Moda":
        df[variables_pregunta_1].apply(obtener_moda),

    "Varianza":
        df[variables_pregunta_1].var(),

    "Desviación estándar":
        df[variables_pregunta_1].std(),

    "Kurtosis":
        df[variables_pregunta_1].kurt()

})


print("Características físicas, orbitales y estelares generales de los exoplanetas registrados \n")
print(estadisticas_generales.round(3))


# Principales características físicas y orbitales de los exoplanetas según su método de descubrimiento



# Variables físicas y orbitales que se compararán
variables_comparacion = [

    "pl_orbper",      # Periodo orbital
    "pl_orbsmax",     # Semieje mayor orbital
    "pl_rade",        # Radio del planeta
    "pl_bmasse",      # Masa del planeta
    "pl_orbeccen",    # Excentricidad orbital
    "pl_eqt"          # Temperatura de equilibrio

]


# Agrupar los exoplanetas según su método de descubrimiento
estadisticas_por_metodo = (

    df.groupby("discoverymethod")[variables_comparacion]

    .agg([
        "count",
        "min",
        "max",
        "mean",
        obtener_moda,
        "var",
        "std",
        obtener_kurtosis
    ])

)


print("\n Principales características físicas y orbitales de los exoplanetas según su método de descubrimiento \n")
print(estadisticas_por_metodo.round(3))

print("\nAnálisis estadístico finalizado.")

