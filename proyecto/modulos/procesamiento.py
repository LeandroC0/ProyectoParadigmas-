"""
Módulo: Procesamiento de Datos

- Detectar automáticamente el tipo de cada columna (numérica, categórica, fecha, booleana)
- Limpieza básica: valores nulos, duplicados, tipos incorrectos.


"""

from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np

TipoColumna = Literal["numerica", "categorica", "fecha", "booleana", "texto_libre"]


@dataclass
class ResultadoProcesamiento:
    dataframe: pd.DataFrame
    tipos_columna: dict[str, TipoColumna]
    reporte_limpieza: dict = field(default_factory=dict)



#DETECCIÓN AUTOMÁTICA DE TIPOS
def detectar_tipos_columna(df: pd.DataFrame) -> dict[str, TipoColumna]:
 
    tipos = {}

    for columna in df.columns:
        serie = df[columna]
        tipos[columna] = _detectar_tipo_individual(serie)

    return tipos


def _detectar_tipo_individual(serie: pd.Series) -> TipoColumna:
    serie_sin_nulos = serie.dropna()

    if serie_sin_nulos.empty:
        return "texto_libre"  # columna totalmente vacía, se maneja aparte

    # --- 1. Booleana ---
    valores_unicos = set(serie_sin_nulos.astype(str).str.lower().unique())
    conjuntos_booleanos = [
        {"true", "false"},
        {"si", "no"},
        {"sí", "no"},
        {"1", "0"},
        {"verdadero", "falso"},
    ]
    if serie.dtype == bool or any(valores_unicos <= s for s in conjuntos_booleanos):
        if len(valores_unicos) <= 2:
            return "booleana"

    # --- 2. Fecha ---
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    if _parece_fecha(serie_sin_nulos):
        return "fecha"

    # --- 3. Numérica ---
    if pd.api.types.is_numeric_dtype(serie):
        return "numerica"
    if _parece_numerica(serie_sin_nulos):
        return "numerica"

    # --- 4. Categórica vs texto libre ---
    n_unicos = serie_sin_nulos.nunique()
    n_total = len(serie_sin_nulos)
    proporcion_unicos = n_unicos / n_total


    if n_unicos <= 20 or proporcion_unicos < 0.5:
        return "categorica"

    return "texto_libre"


def _parece_fecha(serie: pd.Series, muestra: int = 20) -> bool:
    """Intenta parsear una muestra de la columna como fecha."""
    submuestra = serie.astype(str).sample(min(muestra, len(serie)), random_state=1)
    try:
        parseadas = pd.to_datetime(submuestra, errors="coerce", format="mixed")
        # Si al menos el 80% de la muestra parsea correctamente, la consideramos fecha
        return parseadas.notna().mean() >= 0.8
    except Exception:
        return False


def _parece_numerica(serie: pd.Series, muestra: int = 50) -> bool:
    """Detecta números guardados como texto, ej: '1.200.000' o '$500'."""
    submuestra = serie.astype(str).sample(min(muestra, len(serie)), random_state=1)
    convertidos = submuestra.apply(_texto_a_numero)
    return convertidos.notna().mean() >= 0.8


def _texto_a_numero(valor: str) -> float:
    
    if valor is None or str(valor).strip().lower() in ("nan", "none", ""):
        return np.nan

    texto = str(valor).strip()
    texto = pd.Series([texto]).str.replace(r"[$%\s]", "", regex=True).iloc[0]

    tiene_coma = "," in texto
    tiene_punto = "." in texto

    if tiene_coma and tiene_punto:
        texto = texto.replace(".", "").replace(",", ".")
    elif tiene_punto and texto.count(".") > 1:
        texto = texto.replace(".", "")
    elif tiene_coma:
        texto = texto.replace(",", "")

    try:
        return float(texto)
    except ValueError:
        return np.nan




def limpiar_datos(
    df: pd.DataFrame,
    tipos_columna: dict[str, TipoColumna],
    estrategia_nulos: Literal["eliminar_fila", "imputar", "dejar"] = "imputar",
) -> ResultadoProcesamiento:
    """
    Limpia el DataFrame aplicando:
    - Conversión real de tipos según lo detectado.
    - Manejo de duplicados.
    - Manejo de valores nulos.

    Devuelve el DataFrame limpio junto con un reporte de lo que se hizo,
    para que el módulo de interpretación pueda mencionarlo
    (ej: "Se eliminaron 12 filas duplicadas").
    """
    df_limpio = df.copy()
    reporte = {}

    # --- Conversión real de tipos ---
    for columna, tipo in tipos_columna.items():
        if tipo == "numerica" and not pd.api.types.is_numeric_dtype(df_limpio[columna]):
            df_limpio[columna] = _convertir_a_numerica(df_limpio[columna])
        elif tipo == "fecha" and not pd.api.types.is_datetime64_any_dtype(df_limpio[columna]):
            df_limpio[columna] = pd.to_datetime(df_limpio[columna], errors="coerce")
        elif tipo == "booleana":
            df_limpio[columna] = _convertir_a_booleana(df_limpio[columna])

    # --- Duplicados ---
    filas_antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates()
    duplicados_eliminados = filas_antes - len(df_limpio)
    reporte["duplicados_eliminados"] = duplicados_eliminados

    # --- Nulos ---
    nulos_por_columna = df_limpio.isna().sum()
    reporte["nulos_detectados"] = nulos_por_columna[nulos_por_columna > 0].to_dict()

    if estrategia_nulos == "eliminar_fila":
        filas_antes = len(df_limpio)
        df_limpio = df_limpio.dropna()
        reporte["filas_eliminadas_por_nulos"] = filas_antes - len(df_limpio)

    elif estrategia_nulos == "imputar":
        for columna, tipo in tipos_columna.items():
            if columna not in df_limpio.columns:
                continue
            if df_limpio[columna].isna().sum() == 0:
                continue
            if tipo == "numerica":
                valor = df_limpio[columna].median()
                df_limpio[columna] = df_limpio[columna].fillna(valor)
            elif tipo in ("categorica", "booleana", "texto_libre"):
                moda = df_limpio[columna].mode()
                if not moda.empty:
                    df_limpio[columna] = df_limpio[columna].fillna(moda.iloc[0])

 
    reporte["filas_finales"] = len(df_limpio)
    reporte["columnas_finales"] = len(df_limpio.columns)

    return ResultadoProcesamiento(
        dataframe=df_limpio,
        tipos_columna=tipos_columna,
        reporte_limpieza=reporte,
    )


def _convertir_a_numerica(serie: pd.Series) -> pd.Series:
    return serie.astype(str).apply(_texto_a_numero)


def _convertir_a_booleana(serie: pd.Series) -> pd.Series:
    mapa = {
        "true": True, "false": False,
        "si": True, "sí": True, "no": False,
        "1": True, "0": False,
        "verdadero": True, "falso": False,
    }
    return serie.astype(str).str.lower().map(mapa)



def procesar(df: pd.DataFrame, estrategia_nulos: str = "imputar") -> ResultadoProcesamiento:
    tipos = detectar_tipos_columna(df)
    return limpiar_datos(df, tipos, estrategia_nulos=estrategia_nulos)



if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from carga_datos import cargar_archivo

    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo/ventas_ejemplo.csv"
    carga = cargar_archivo(ruta)

    if not carga.exito:
        print("Error al cargar:", carga.errores)
    else:
        resultado = procesar(carga.dataframe)
        print("Tipos detectados:")
        for col, tipo in resultado.tipos_columna.items():
            print(f"  {col}: {tipo}")

        print("\nReporte de limpieza:")
        for k, v in resultado.reporte_limpieza.items():
            print(f"  {k}: {v}")

        print("\nVista previa de datos limpios:")
        print(resultado.dataframe.head())
