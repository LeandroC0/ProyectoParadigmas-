"""
 Análisis Exploratorio de Datos 

Este módulo recibe el DataFrame ya limpio que entrega `procesamiento.py`,
junto con el diccionario `tipos_columna` que ese mismo módulo detecta
(numerica, categorica, fecha, booleana, texto_libre).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Any
import pandas as pd
import numpy as np

TipoColumna = Literal["numerica", "categorica", "fecha", "booleana", "texto_libre"]

# A partir de este valor absoluto una correlación se considera digna de mención
UMBRAL_CORRELACION_RELEVANTE = 0.3


# ESTRUCTURAS DE RESULTADO

@dataclass
class EstadisticasNumericas:
    columna: str
    conteo: int
    nulos: int
    media: float
    mediana: float
    desviacion_estandar: float
    minimo: float
    maximo: float
    q1: float
    q3: float
    rango_intercuartilico: float
    asimetria: float
    curtosis: float
    coeficiente_variacion: Optional[float]
    valores_atipicos_iqr: int
    interpretacion_distribucion: str


@dataclass
class EstadisticasCategoricas:
    columna: str
    conteo: int
    nulos: int
    valores_unicos: int
    moda: Any
    frecuencia_moda: int
    porcentaje_moda: float
    top_categorias: dict = field(default_factory=dict)


@dataclass
class ParCorrelacion:
    columna_a: str
    columna_b: str
    coeficiente: float
    fuerza: str
    direccion: str  # "positiva" o "negativa"


@dataclass
class ResultadoAnalisisExploratorio:
    estadisticas_numericas: dict[str, EstadisticasNumericas] = field(default_factory=dict)
    estadisticas_categoricas: dict[str, EstadisticasCategoricas] = field(default_factory=dict)
    matriz_correlacion: Optional[pd.DataFrame] = None
    correlaciones_relevantes: list[ParCorrelacion] = field(default_factory=list)
    resumen_general: dict = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)


# ESTADÍSTICAS NUMÉRICAS

def calcular_estadisticas_numericas(
    df: pd.DataFrame, tipos_columna: dict[str, TipoColumna]
) -> dict[str, EstadisticasNumericas]:
    """Calcula estadísticas descriptivas para cada columna numérica."""
    resultado = {}
    columnas_numericas = [
        c for c, t in tipos_columna.items() if t == "numerica" and c in df.columns
    ]

    for columna in columnas_numericas:
        serie_completa = df[columna]
        serie = serie_completa.dropna()

        if serie.empty:
            continue

        # Regla estándar de Tukey: outlier = fuera de [Q1-1.5*IQR, Q3+1.5*IQR]
        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = int(((serie < limite_inferior) | (serie > limite_superior)).sum())

        media = float(serie.mean())
        desviacion = float(serie.std()) if len(serie) > 1 else 0.0
        cv = (desviacion / media) if media != 0 else None  # evita división por cero en columnas centradas en 0
        asimetria = float(serie.skew()) if len(serie) > 2 else 0.0  # skew necesita 3+ datos para ser válido
        curtosis = float(serie.kurt()) if len(serie) > 3 else 0.0  # kurtosis necesita 4+ datos para ser válida

        resultado[columna] = EstadisticasNumericas(
            columna=columna,
            conteo=int(serie.count()),
            nulos=int(serie_completa.isna().sum()),
            media=media,
            mediana=float(serie.median()),
            desviacion_estandar=desviacion,
            minimo=float(serie.min()),
            maximo=float(serie.max()),
            q1=q1,
            q3=q3,
            rango_intercuartilico=iqr,
            asimetria=asimetria,
            curtosis=curtosis,
            coeficiente_variacion=float(cv) if cv is not None else None,
            valores_atipicos_iqr=outliers,
            interpretacion_distribucion=_interpretar_asimetria(asimetria),
        )

    return resultado


def _interpretar_asimetria(asimetria: float) -> str:
    """Traduce el coeficiente de asimetría a una descripción legible."""
    # Umbrales convencionales de estadística descriptiva: |skew|>1 fuerte, >0.5 moderado
    if asimetria > 1:
        return "fuertemente sesgada a la derecha (cola de valores altos)"
    elif asimetria > 0.5:
        return "moderadamente sesgada a la derecha"
    elif asimetria < -1:
        return "fuertemente sesgada a la izquierda (cola de valores bajos)"
    elif asimetria < -0.5:
        return "moderadamente sesgada a la izquierda"
    else:
        return "aproximadamente simétrica"


# ESTADÍSTICAS CATEGÓRICAS

def calcular_estadisticas_categoricas(
    df: pd.DataFrame, tipos_columna: dict[str, TipoColumna], top_n: int = 5
) -> dict[str, EstadisticasCategoricas]:
    """Calcula frecuencias y moda para columnas categóricas y booleanas."""
    resultado = {}
    columnas_categoricas = [
        c for c, t in tipos_columna.items()
        if t in ("categorica", "booleana") and c in df.columns
    ]

    for columna in columnas_categoricas:
        serie_completa = df[columna]
        serie = serie_completa.dropna()

        if serie.empty:
            continue

        conteos = serie.value_counts()
        moda = conteos.index[0]
        frecuencia_moda = int(conteos.iloc[0])
        total = int(serie.count())

        resultado[columna] = EstadisticasCategoricas(
            columna=columna,
            conteo=total,
            nulos=int(serie_completa.isna().sum()),
            valores_unicos=int(serie.nunique()),
            moda=moda,
            frecuencia_moda=frecuencia_moda,
            porcentaje_moda=round(frecuencia_moda / total * 100, 2),
            top_categorias={
                str(k): int(v) for k, v in conteos.head(top_n).items()
            },
        )

    return resultado


# CORRELACIONES

def calcular_matriz_correlacion(
    df: pd.DataFrame, tipos_columna: dict[str, TipoColumna], metodo: str = "pearson"
) -> Optional[pd.DataFrame]:
    """
    Calcula la matriz de correlación entre columnas numéricas.
    Devuelve None si hay menos de 2 columnas numéricas.
    """
    columnas_numericas = [
        c for c, t in tipos_columna.items() if t == "numerica" and c in df.columns
    ]

    if len(columnas_numericas) < 2:
        return None

    return df[columnas_numericas].corr(method=metodo)


def extraer_correlaciones_relevantes(
    matriz_correlacion: Optional[pd.DataFrame],
    umbral: float = UMBRAL_CORRELACION_RELEVANTE,
) -> list[ParCorrelacion]:
    """
    Filtra pares de columnas con correlación relevante, sin duplicados ni diagonal.
    """
    if matriz_correlacion is None:
        return []

    pares = []
    columnas = matriz_correlacion.columns.tolist()

    # columnas[i+1:] recorre solo el triángulo superior: evita diagonal (col consigo misma) y pares duplicados (A-B / B-A)
    for i, col_a in enumerate(columnas):
        for col_b in columnas[i + 1:]:
            coeficiente = matriz_correlacion.loc[col_a, col_b]

            if pd.isna(coeficiente):
                continue

            if abs(coeficiente) >= umbral:
                pares.append(
                    ParCorrelacion(
                        columna_a=col_a,
                        columna_b=col_b,
                        coeficiente=round(float(coeficiente), 4),
                        fuerza=_interpretar_fuerza_correlacion(abs(coeficiente)),
                        direccion="positiva" if coeficiente > 0 else "negativa",
                    )
                )

    pares.sort(key=lambda p: abs(p.coeficiente), reverse=True)
    return pares


def _interpretar_fuerza_correlacion(valor_absoluto: float) -> str:
    if valor_absoluto >= 0.7:
        return "muy fuerte"
    elif valor_absoluto >= 0.5:
        return "fuerte"
    elif valor_absoluto >= 0.3:
        return "moderada"
    else:
        return "débil"


# RESUMEN GENERAL

def _construir_resumen_general(
    df: pd.DataFrame, tipos_columna: dict[str, TipoColumna]
) -> dict:
    conteo_tipos = {}
    for tipo in tipos_columna.values():
        conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + 1

    total_celdas = df.shape[0] * df.shape[1]
    total_nulos = int(df.isna().sum().sum())

    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "conteo_por_tipo": conteo_tipos,
        "porcentaje_nulos_global": round(total_nulos / total_celdas * 100, 2) if total_celdas else 0.0,  # guard: df con 0 celdas
        "filas_duplicadas": int(df.duplicated().sum()),
        "memoria_uso_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),  # deep=True cuenta el peso real de strings/objetos
    }


# FUNCIÓN PRINCIPAL

def analizar(
    df: pd.DataFrame,
    tipos_columna: dict[str, TipoColumna],
    metodo_correlacion: str = "pearson",
    umbral_correlacion: float = UMBRAL_CORRELACION_RELEVANTE,
) -> ResultadoAnalisisExploratorio:
    """
    Recibe el DataFrame limpio y los tipos
    detectados (salida de `procesamiento.procesar`) y devuelve un
    `ResultadoAnalisisExploratorio` con todo lo necesario para que se hagan los demas modulos
    """
    advertencias = []

    if df.empty:
        advertencias.append("El DataFrame está vacío, no se puede analizar.")
        return ResultadoAnalisisExploratorio(advertencias=advertencias)

    estadisticas_numericas = calcular_estadisticas_numericas(df, tipos_columna)
    estadisticas_categoricas = calcular_estadisticas_categoricas(df, tipos_columna)

    if not estadisticas_numericas:
        advertencias.append("No se encontraron columnas numéricas para analizar.")

    matriz_correlacion = calcular_matriz_correlacion(df, tipos_columna, metodo=metodo_correlacion)
    if matriz_correlacion is None and len(estadisticas_numericas) >= 1:
        advertencias.append(
            "Se necesitan al menos 2 columnas numéricas para calcular correlaciones."
        )

    correlaciones_relevantes = extraer_correlaciones_relevantes(
        matriz_correlacion, umbral=umbral_correlacion
    )

    resumen_general = _construir_resumen_general(df, tipos_columna)

    return ResultadoAnalisisExploratorio(
        estadisticas_numericas=estadisticas_numericas,
        estadisticas_categoricas=estadisticas_categoricas,
        matriz_correlacion=matriz_correlacion,
        correlaciones_relevantes=correlaciones_relevantes,
        resumen_general=resumen_general,
        advertencias=advertencias,
    )


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from carga_datos import cargar_archivo
    from procesamiento import procesar

    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo/ventas_ejemplo.csv"
    carga = cargar_archivo(ruta)

    if not carga.exito:
        print("Error al cargar:", carga.errores)
    else:
        procesado = procesar(carga.dataframe)
        resultado = analizar(procesado.dataframe, procesado.tipos_columna)

        print("=== Resumen general ===")
        for k, v in resultado.resumen_general.items():
            print(f"  {k}: {v}")

        print("\n=== Estadísticas numéricas ===")
        for columna, stats in resultado.estadisticas_numericas.items():
            print(f"\n  {columna}")
            print(f"    media: {stats.media:.2f} | mediana: {stats.mediana:.2f} | "
                  f"desv. estándar: {stats.desviacion_estandar:.2f}")
            print(f"    min: {stats.minimo:.2f} | max: {stats.maximo:.2f} | "
                  f"outliers (IQR): {stats.valores_atipicos_iqr}")
            print(f"    distribución: {stats.interpretacion_distribucion}")

        print("\n=== Estadísticas categóricas ===")
        for columna, stats in resultado.estadisticas_categoricas.items():
            print(f"\n  {columna}")
            print(f"    valores únicos: {stats.valores_unicos} | "
                  f"moda: {stats.moda} ({stats.porcentaje_moda}%)")
            print(f"    top categorías: {stats.top_categorias}")

        print("\n=== Correlaciones relevantes ===")
        if resultado.correlaciones_relevantes:
            for par in resultado.correlaciones_relevantes:
                print(f"  {par.columna_a} <-> {par.columna_b}: "
                      f"{par.coeficiente} ({par.fuerza}, {par.direccion})")
        else:
            print("  No se encontraron correlaciones relevantes.")

        if resultado.advertencias:
            print("\n=== Advertencias ===")
            for a in resultado.advertencias:
                print(f"  - {a}")