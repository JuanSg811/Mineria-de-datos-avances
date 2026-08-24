import re
import pandas as pd
import numpy as np

INPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\NASA_Exoplanet_Archive_Data.csv"
OUTPUT_PATH = r"C:\Users\lego8\OneDrive\Escritorio\Tarea Mineria de datos\Dataset_Limpiado.csv"

COLUMNAS_RELEVANTES = [
    "pl_name", "hostname", "default_flag",
    "disc_year", "pl_pubdate", "rowupdate", "releasedate",
    "discoverymethod", "disc_facility", "ttv_flag",
    "pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse",
    "pl_orbeccen", "pl_eqt", "pl_insol",
    "st_spectype", "st_teff", "st_rad", "st_mass", "st_met",
    "sy_snum", "sy_pnum", "sy_dist", "sy_vmag",
]

# Columnas de fecha 
COLUMNAS_FECHA_YMD = ["rowupdate", "releasedate"]  
COLUMNAS_FECHA_YM = ["pl_pubdate"]                  

# Columnas que deben ser numéricas
COLUMNAS_NUMERICAS = [
    "disc_year", "pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse",
    "pl_orbeccen", "pl_eqt", "pl_insol", "st_teff", "st_rad",
    "st_mass", "st_met", "sy_snum", "sy_pnum", "sy_dist", "sy_vmag",
]

# Columnas categóricas / de texto
COLUMNAS_CATEGORICAS = [
    "pl_name", "hostname", "discoverymethod", "disc_facility", "st_spectype",
]

UMBRAL_AVISO_NULOS = 0.5

def cargar_datos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[1] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df

def seleccionar_columnas(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    columnas_existentes = [c for c in columnas if c in df.columns]
    faltantes = set(columnas) - set(columnas_existentes)
    if faltantes:
        print(f"    Aviso: columnas no encontradas en el dataset: {faltantes}")
    df = df[columnas_existentes].copy()
    print(f"[2] Columnas seleccionadas: {df.shape[1]}")
    return df

def deduplicar_por_planeta(df: pd.DataFrame) -> pd.DataFrame:
    if "default_flag" in df.columns:
        antes = df.shape[0]
        df = df[df["default_flag"] == 1].drop(columns=["default_flag"])
        print(f"[3] Deduplicado por planeta: {antes} -> {df.shape[0]} filas "
              f"(1 registro por planeta, usando default_flag=1)")
    return df


def limpiar_numero_texto(valor):
    """Convierte un valor potencialmente en texto ('$1,234', '56%', ' 12 ')
    a float. Devuelve np.nan si no se puede convertir."""
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip()
    # quita símbolos comunes: $, comas de miles, %, espacios
    texto = re.sub(r"[\$,%\s]", "", texto)
    if texto in ("", "-", "nan", "None", "N/A"):
        return np.nan
    try:
        return float(texto)
    except ValueError:
        return np.nan


def convertir_columnas_numericas(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    for col in columnas:
        if col not in df.columns:
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(limpiar_numero_texto)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[4] Columnas numéricas normalizadas: {len(columnas)}")
    return df

def convertir_fechas(df: pd.DataFrame, columnas_ymd: list, columnas_ym: list) -> pd.DataFrame:
    for col in columnas_ymd:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    for col in columnas_ym:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m", errors="coerce")
    total = len(columnas_ymd) + len(columnas_ym)
    print(f"[5] Columnas de fecha convertidas a datetime: {total}")
    return df

def limpiar_texto_categorico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    for col in columnas:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        # valores vacíos tras limpiar quedan como nulos reales
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    print(f"[6] Columnas de texto limpiadas (espacios, formato): {len(columnas)}")
    return df

def reportar_nulos(df: pd.DataFrame) -> pd.Series:
    porcentaje_nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    print("\n[7] % de valores nulos por columna:")
    print(porcentaje_nulos.round(1).to_string())
    return porcentaje_nulos


def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    numericas = df.select_dtypes(include=[np.number]).columns
    for col in numericas:
        if df[col].isna().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    categoricas = df.select_dtypes(include=["string", "object"]).columns
    for col in categoricas:
        if df[col].isna().any():
            df[col] = df[col].fillna("Desconocido")

    print(f"[8] Valores vacíos tratados: numéricas -> mediana, "
          f"categóricas -> 'Desconocido', fechas -> sin imputar")
    return df

def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    if "disc_year" in df.columns:
        df["disc_decada"] = (df["disc_year"] // 10 * 10).astype("Int64")

    if "pl_rade" in df.columns:
        condiciones = [
            df["pl_rade"] < 1.5,
            df["pl_rade"] < 2.0,
            df["pl_rade"] < 6.0,
        ]
        categorias = ["Rocoso", "Super-Tierra", "Mini-Neptuno"]
        df["pl_categoria"] = np.select(condiciones, categorias, default="Gigante")

    print("[9] Variables derivadas creadas: disc_decada, pl_categoria")
    return df

def main():
    df = cargar_datos(INPUT_PATH)
    df = seleccionar_columnas(df, COLUMNAS_RELEVANTES)
    df = deduplicar_por_planeta(df)
    df = convertir_columnas_numericas(df, COLUMNAS_NUMERICAS)
    df = convertir_fechas(df, COLUMNAS_FECHA_YMD, COLUMNAS_FECHA_YM)
    df = limpiar_texto_categorico(df, COLUMNAS_CATEGORICAS)
    reportar_nulos(df)
    df = tratar_nulos(df)
    df = crear_variables_derivadas(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset limpio guardado en: {OUTPUT_PATH}")
    print(f"Forma final: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nVista previa:")
    print(df.head())

    return df


if __name__ == "__main__":
    main()
