"""
Carga de Datos

- Importar archivos CSV y Excel.
- Validar que el archivo tenga formato y estructura correctos.
- Devolver un DataFrame de pandas listo para el siguiente módulo (procesamiento).
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import os


@dataclass
class ResultadoCarga:
    exito: bool
    dataframe: Optional[pd.DataFrame] = None
    nombre_archivo: str = ""
    filas: int = 0
    columnas: int = 0
    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)


# Extensiones que el sistema sabe procesar
EXTENSIONES_VALIDAS = {".csv", ".xlsx", ".xls"}

# Límite para evitar que alguien suba un archivo gigante

MAX_FILAS_RECOMENDADO = 500_000


def cargar_archivo(ruta_o_buffer, nombre_archivo: str = None) -> ResultadoCarga:
  
    if nombre_archivo is None:
        nombre_archivo = ruta_o_buffer if isinstance(ruta_o_buffer, str) else getattr(
            ruta_o_buffer, "name", "archivo_desconocido"
        )

    extension = os.path.splitext(nombre_archivo)[1].lower()

    if extension not in EXTENSIONES_VALIDAS:
        return ResultadoCarga(
            exito=False,
            nombre_archivo=nombre_archivo,
            errores=[
                f"Extensión '{extension}' no soportada. "
                f"Formatos válidos: {', '.join(EXTENSIONES_VALIDAS)}"
            ],
        )

    try:
        if extension == ".csv":
            df = _leer_csv(ruta_o_buffer)
        else:
            df = pd.read_excel(ruta_o_buffer)
    except Exception as e:
        return ResultadoCarga(
            exito=False,
            nombre_archivo=nombre_archivo,
            errores=[f"No se pudo leer el archivo: {e}"],
        )

    resultado = _validar_estructura(df, nombre_archivo)
    return resultado


def _leer_csv(ruta_o_buffer) -> pd.DataFrame:

    try:
        # sep=None + engine='python' activa la detección automática de separador
        return pd.read_csv(ruta_o_buffer, sep=None, engine="python")
    except Exception:
        # Si falla la detección automática, reintentamos con coma por defecto
        if hasattr(ruta_o_buffer, "seek"):
            ruta_o_buffer.seek(0)
        return pd.read_csv(ruta_o_buffer)


def _validar_estructura(df: pd.DataFrame, nombre_archivo: str) -> ResultadoCarga:
 
    errores = []
    advertencias = []

    # 1. El archivo no puede estar vacío
    if df.empty:
        errores.append("El archivo no contiene datos.")

    # 2. Debe tener al menos una columna con nombre válido (no "Unnamed: 0")
    columnas_sin_nombre = [c for c in df.columns if str(c).startswith("Unnamed")]
    if columnas_sin_nombre:
        advertencias.append(
            f"Se detectaron {len(columnas_sin_nombre)} columna(s) sin encabezado. "
            "Verifica que el archivo tenga una fila de títulos."
        )

    # 3. Debe tener al menos 2 columnas para que el análisis tenga sentido
    if df.shape[1] < 2:
        errores.append("El archivo debe tener al menos 2 columnas para poder analizarse.")

    # 4. Debe tener al menos algunas filas de datos
    if df.shape[0] < 3:
        errores.append("El archivo debe tener al menos 3 filas de datos.")

    # 5. Advertencia si el archivo es muy grande
    if df.shape[0] > MAX_FILAS_RECOMENDADO:
        advertencias.append(
            f"El archivo tiene {df.shape[0]:,} filas. "
            "El procesamiento podría tardar más de lo esperado."
        )

    # 6. Nombres de columnas duplicados rompen varios análisis después
    duplicadas = df.columns[df.columns.duplicated()].tolist()
    if duplicadas:
        errores.append(f"Nombres de columna duplicados: {duplicadas}")

    exito = len(errores) == 0

    return ResultadoCarga(
        exito=exito,
        dataframe=df if exito else None,
        nombre_archivo=nombre_archivo,
        filas=df.shape[0],
        columnas=df.shape[1],
        errores=errores,
        advertencias=advertencias,
    )



if __name__ == "__main__":
    import sys

    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo/ventas_ejemplo.csv"
    resultado = cargar_archivo(ruta)

    print(f"Archivo: {resultado.nombre_archivo}")
    print(f"Éxito: {resultado.exito}")
    print(f"Filas x Columnas: {resultado.filas} x {resultado.columnas}")

    if resultado.advertencias:
        print("\nAdvertencias:")
        for a in resultado.advertencias:
            print(f"  - {a}")

    if resultado.errores:
        print("\nErrores:")
        for e in resultado.errores:
            print(f"  - {e}")
    elif resultado.exito:
        print("\nVista previa:")
        print(resultado.dataframe.head())
