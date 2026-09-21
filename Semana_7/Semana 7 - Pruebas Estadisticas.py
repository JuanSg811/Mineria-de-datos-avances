import pandas as pd
import numpy as np

from scipy import stats
from itertools import combinations

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Dataset_Limpiado.csv"

df = pd.read_csv(INPUT_PATH)

print("PRUEBAS ESTADÍSTICAS - EXOPLANETAS")

print(f"Registros: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")


variable = "pl_rade"

grupo = "discoverymethod"

datos = df[[grupo, variable]].dropna()

print("\nDatos utilizados:")
print(f"Registros válidos: {len(datos)}")

#Aquí filtro los metodos con suficientes observaciones
MINIMO_REGISTROS = 30

conteo = datos[grupo].value_counts()

metodos_validos = conteo[
    conteo >= MINIMO_REGISTROS
].index


datos_filtrados = datos[
    datos[grupo].isin(metodos_validos)
]


print("\nMétodos incluidos en el análisis:")

print(
    datos_filtrados[grupo]
    .value_counts()
)

grupos = []

nombres_grupos = []

for metodo in metodos_validos:

    valores = datos_filtrados.loc[
        datos_filtrados[grupo] == metodo,
        variable
    ]

    grupos.append(valores)

    nombres_grupos.append(metodo)


estadisticas = (
    datos_filtrados
    .groupby(grupo)[variable]
    .agg(
        conteo="count",
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        desviacion_estandar="std"
    )
)

print("\n")
print("ESTADÍSTICA DESCRIPTIVA")

print(estadisticas.round(3))

# #Prueba ANOVA
resultado_anova = stats.f_oneway(*grupos)

F = resultado_anova.statistic
p = resultado_anova.pvalue


print("\n")
print("ANOVA DE UNA VÍA")

print(f"F = {F:.4f}")
print(f"p-valor = {p:.6f}")


ALPHA = 0.05


if p < ALPHA:

    print("\nResultado:")
    print("Se rechaza H0.")
    print(
        "Existen diferencias estadísticamente significativas "
        "entre al menos dos métodos de descubrimiento."
    )

else:

    print("\nResultado:")
    print("No se rechaza H0.")
    print(
        "No se encontraron diferencias estadísticamente significativas "
        "entre las medias de los métodos de descubrimiento."
    )

#Kruskal-Wallis.
resultado_kw = stats.kruskal(*grupos)

H = resultado_kw.statistic
p_kw = resultado_kw.pvalue


print("\n")
print("PRUEBA DE KRUSKAL-WALLIS")

print(f"H = {H:.4f}")
print(f"p-valor = {p_kw:.6f}")


if p_kw < ALPHA:

    print("\nResultado:")
    print("Se rechaza H0.")
    print(
        "Existen diferencias estadísticamente significativas "
        "entre al menos dos grupos."
    )

else:

    print("\nResultado:")
    print("No se rechaza H0.")
    print(
        "No se encontraron diferencias estadísticamente significativas "
        "entre los grupos."
    )



# 9. PRUEBAS T ENTRE PARES DE MÉTODOS

print("\n")
print("PRUEBAS T ENTRE PARES DE MÉTODOS")


metodos_prueba_t = (
    datos_filtrados[grupo]
    .value_counts()
    .head(2)
    .index
)


if len(metodos_prueba_t) == 2:

    metodo_1 = metodos_prueba_t[0]
    metodo_2 = metodos_prueba_t[1]

    grupo_1 = datos_filtrados.loc[
        datos_filtrados[grupo] == metodo_1,
        variable
    ]

    grupo_2 = datos_filtrados.loc[
        datos_filtrados[grupo] == metodo_2,
        variable
    ]

    resultado_t = stats.ttest_ind(
        grupo_1,
        grupo_2,
        equal_var=False
    )

    print(f"\nComparación:")
    print(f"{metodo_1} vs {metodo_2}")

    print(f"t = {resultado_t.statistic:.4f}")
    print(f"p-valor = {resultado_t.pvalue:.6f}")

    if resultado_t.pvalue < ALPHA:

        print("\nResultado:")
        print("Se rechaza H0.")
        print(
            "Existe una diferencia estadísticamente significativa "
            "entre ambos métodos."
        )

    else:

        print("\nResultado:")
        print("No se rechaza H0.")
        print(
            "No se encontró una diferencia estadísticamente significativa "
            "entre ambos métodos."

        )

else:

    print("No hay suficientes grupos para realizar la prueba t.")
