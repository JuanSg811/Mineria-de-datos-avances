"""
Objetivo:
Analizar la relación lineal entre la masa y el radio de los exoplanetas
registrados, mediante el coeficiente de correlación de Pearson y un modelo
de regresión lineal, usando R^2 como métrica de capacidad explicativa.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\NASA_Exoplanet_Archive_Data.csv" 
OUTPUT_DIR = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Semana_9"
ALPHA = 0.05

plt.rcParams["figure.dpi"] = 110
COLOR_PUNTOS = "#2c6e91"
COLOR_RECTA = "#d1495b"

df = pd.read_csv(INPUT_PATH)
df = df[df["default_flag"] == 1]  # 1 fila por planeta (dato oficial)

datos = df[["pl_name", "discoverymethod", "pl_rade", "pl_bmasse"]].dropna(
    subset=["pl_rade", "pl_bmasse"]
)
datos = datos[(datos["pl_rade"] > 0) & (datos["pl_bmasse"] > 0)]  # log requiere >0

print(f"Planetas con masa Y radio medidos (sin imputar): {len(datos)} "
      f"de {df.shape[0]} planetas totales")


datos["log_rade"] = np.log10(datos["pl_rade"])
datos["log_bmasse"] = np.log10(datos["pl_bmasse"])


# 1. CORRELACION DE PEARSON
print("\n" + "=" * 90)
print("1. CORRELACION DE PEARSON")
print("=" * 90)

r_original, p_original = stats.pearsonr(datos["pl_rade"], datos["pl_bmasse"])
print(f"\nEscala original:  r = {r_original:.3f}   p-valor = {p_original:.2e}")

r_log, p_log = stats.pearsonr(datos["log_rade"], datos["log_bmasse"])
print(f"Escala log-log:    r = {r_log:.3f}   p-valor = {p_log:.2e}  <- se usa para el modelo")

print(f"\n>> En escala original la relación es débil (r={r_original:.2f}) porque "
      f"unos pocos gigantes gaseosos muy masivos distorsionan la escala lineal.")
print(f">> En escala log-log la correlación es fuerte (r={r_log:.2f}), consistente "
      f"con la relación masa-radio conocida en física planetaria (ley de potencia).")

# 2. MODELO DE REGRESION LINEAL (en escala log-log)

print("\n" + "=" * 90)
print("2. MODELO DE REGRESION LINEAL: log10(masa) ~ log10(radio)")
print("=" * 90)

resultado = stats.linregress(datos["log_rade"], datos["log_bmasse"])
pendiente, intercepto, r_valor, p_valor, error_estandar = resultado
r2 = r_valor ** 2

print(f"""
Ecuación del modelo:  log10(masa) = {pendiente:.3f} * log10(radio) + {intercepto:.3f}
Equivalente en escala original:  masa = 10^{intercepto:.3f} * radio^{pendiente:.3f}

Pendiente            : {pendiente:.4f}  (error estándar: {error_estandar:.4f})
Intercepto           : {intercepto:.4f}
Coeficiente r        : {r_valor:.4f}
R^2                  : {r2:.4f}
p-valor              : {p_valor:.2e}
""")
print(f">> El modelo explica el {r2*100:.1f}% de la varianza de log10(masa) "
      f"a partir de log10(radio).")
print(f">> {'SI' if p_valor < ALPHA else 'NO'} hay una relación lineal "
      f"estadísticamente significativa (p {'<' if p_valor < ALPHA else '>='} {ALPHA}).")

# Predicciones y residuales (para la gráfica de diagnóstico)
datos["log_bmasse_predicho"] = intercepto + pendiente * datos["log_rade"]
datos["residual"] = datos["log_bmasse"] - datos["log_bmasse_predicho"]


# 3. GRAFICAS

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- 3.1 Dispersión en escala original -------------------------------------
axes[0].scatter(datos["pl_rade"], datos["pl_bmasse"], alpha=0.4, s=15, color=COLOR_PUNTOS)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_title(f"Masa vs. radio (escala log, sin ajuste)\nr = {r_original:.3f}")
axes[0].set_xlabel("Radio (radios terrestres)")
axes[0].set_ylabel("Masa (masas terrestres)")
axes[0].grid(alpha=0.3, which="both")

# --- 3.2 Regresión en escala log-log con la recta ajustada -----------------
axes[1].scatter(datos["log_rade"], datos["log_bmasse"], alpha=0.4, s=15, color=COLOR_PUNTOS,
                 label="Planetas")
x_linea = np.linspace(datos["log_rade"].min(), datos["log_rade"].max(), 100)
y_linea = intercepto + pendiente * x_linea
axes[1].plot(x_linea, y_linea, color=COLOR_RECTA, linewidth=2,
             label=f"Ajuste lineal (R²={r2:.3f})")
axes[1].set_title("Modelo de regresión lineal\nlog10(masa) ~ log10(radio)")
axes[1].set_xlabel("log10(radio)")
axes[1].set_ylabel("log10(masa)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- 3.3 Gráfica de residuales (diagnóstico del modelo) --------------------
axes[2].scatter(datos["log_bmasse_predicho"], datos["residual"], alpha=0.4, s=15,
                 color=COLOR_PUNTOS)
axes[2].axhline(0, color=COLOR_RECTA, linewidth=2)
axes[2].set_title("Residuales del modelo")
axes[2].set_xlabel("log10(masa) predicho")
axes[2].set_ylabel("Residual (real - predicho)")
axes[2].grid(alpha=0.3)

fig.tight_layout()
ruta_salida = f"{OUTPUT_DIR}/modelo_lineal_masa_radio.png"
fig.savefig(ruta_salida)
plt.close(fig)
print(f"\nGráfica guardada: {ruta_salida}")
