from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def construir_resumen(
    resultado_ia: dict,
    resultado_exploratorio: Any = None,
    reporte_limpieza: Optional[dict] = None,
    nombre_dataset: str = "ventas_ejemplo.csv",
    max_columnas_numericas: int = 6,
    max_categorias: int = 3,
    max_correlaciones: int = 5,
) -> dict:

    # ==============================
    # 1. Se arma la base del resumen
    # ==============================


    resumen = {
        "dataset": nombre_dataset
    }

    # ==============================
    # 2. Se agrega el reporte de limpieza (si existe)
    # ==============================

    if reporte_limpieza:

        resumen["limpieza"] = {
            "filas_finales": reporte_limpieza.get("filas_finales"),
            "columnas_finales": reporte_limpieza.get("columnas_finales"),
            "duplicados_eliminados": reporte_limpieza.get("duplicados_eliminados"),
            "nulos_detectados": reporte_limpieza.get("nulos_detectados"),
        }

    # ==============================
    # 3. Se agrega el análisis exploratorio , si se pasó
    # ==============================

    if resultado_exploratorio is not None:

        resumen["exploratorio"] = _resumir_exploratorio(
            resultado_exploratorio,
            max_columnas_numericas,
            max_categorias,
            max_correlaciones
        )

    # ==============================
    # 4. Se agrega el clustering y los outliers 
    # ==============================

    resumen["clustering"] = _resumir_clustering(resultado_ia)

    resumen["outliers"] = _resumir_outliers(resultado_ia)

    return resumen


def _resumir_exploratorio(resultado_exploratorio, max_num, max_cat, max_corr) -> dict:


    r = asdict(resultado_exploratorio) if is_dataclass(resultado_exploratorio) else resultado_exploratorio

    out = {}

    if r.get("resumen_general"):
        out["resumen_general"] = r["resumen_general"]

# Se priorizan las columnas numéricas que presentan más valores atípicos
# o variaciones importantes, y se seleccionan solo las más relevantes
# para evitar analizar datos que no aporten mucha información.

    numericas = r.get("estadisticas_numericas", {}) or {}

    numericas_ordenadas = sorted(
        numericas.values(),
        key=lambda s: (s.get("valores_atipicos_iqr", 0), abs(s.get("asimetria", 0))),
        reverse=True,
    )[:max_num]

    out["estadisticas_numericas"] = [
        {
            "columna": s["columna"],
            "media": round(s["media"], 2),
            "mediana": round(s["mediana"], 2),
            "desviacion_estandar": round(s["desviacion_estandar"], 2),
            "distribucion": s["interpretacion_distribucion"],
            "outliers_iqr": s["valores_atipicos_iqr"],
        }
        for s in numericas_ordenadas
    ]

    categoricas = r.get("estadisticas_categoricas", {}) or {}

    out["estadisticas_categoricas"] = [
        {
            "columna": s["columna"],
            "valores_unicos": s["valores_unicos"],
            "moda": s["moda"],
            "porcentaje_moda": s["porcentaje_moda"],
        }
        for s in list(categoricas.values())[:max_cat]
    ]

    correlaciones = r.get("correlaciones_relevantes", []) or []

    out["correlaciones_relevantes"] = [
        {
            "columnas": f"{c['columna_a']} - {c['columna_b']}",
            "coeficiente": c["coeficiente"],
            "fuerza": c["fuerza"],
            "direccion": c["direccion"],
        }
        for c in correlaciones[:max_corr]
    ]

    if r.get("advertencias"):
        out["advertencias"] = r["advertencias"]

    return out


def _resumir_clustering(resultado_ia: dict) -> dict:

    resumen_clusters_df = resultado_ia["resumen_clusters"]

    return {

        "numero_clusters":
            resultado_ia["mejor_numero_clusters"],

        "calidad_clustering":
            _interpretar_silhouette(resultado_ia["mejor_silhouette"]),

        "silhouette_score":
            round(float(resultado_ia["mejor_silhouette"]), 3),

        "variables_usadas":
            resultado_ia["columnas_utilizadas"],

        # to_dict("records") entrega una lista de dicts, uno por
        # cluster, ya son promedios agregados, no datos individuales
        # de personas

        "resumen_por_cluster":
            resumen_clusters_df.to_dict("records"),
    }


def _interpretar_silhouette(score: float) -> str:

# Se interpreta el Silhouette Score según su valor para indicar
# qué tan bien quedaron definidos los grupos.

    if score >= 0.7:
        return "clusters muy bien definidos y separados"
    elif score >= 0.5:
        return "clusters razonablemente bien definidos"
    elif score >= 0.25:
        return "clusters con separación débil, hay solapamiento"
    else:
        return "clusters poco claros, los grupos no están bien diferenciados"


def _resumir_outliers(resultado_ia: dict) -> dict:

    outliers_df = resultado_ia["outliers"]
    columnas = resultado_ia["columnas_utilizadas"]
    cantidad = resultado_ia["cantidad_outliers"]

    resultado = {
        "cantidad_outliers": cantidad,
        "total_registros": len(resultado_ia["dataframe"]),
    }

    if cantidad > 0:

# Se comparan los valores atípicos con el promedio general
# sin mostrar información específica de cada registro.

        promedios_outliers = outliers_df[columnas].mean().round(2).to_dict()
        promedios_generales = resultado_ia["dataframe"][columnas].mean().round(2).to_dict()

        resultado["comparacion_outliers_vs_general"] = {
            columna: {
                "promedio_outliers": promedios_outliers[columna],
                "promedio_general": promedios_generales[columna],
            }
            for columna in columnas
        }

    return resultado


def construir_prompt(resumen: dict) -> str:

    # Se convierte el resumen a texto JSON para insertarlo directo
    # en el prompt

    datos_json = json.dumps(resumen, ensure_ascii=False, indent=2, default=str)

    prompt = f"""Eres un analista de datos que explica resultados a una persona
SIN conocimientos técnicos (no sabe qué es un "silhouette score" ni un
"coeficiente de correlación").

A continuación tienes los resultados de un análisis exploratorio y de
Inteligencia Artificial (clustering y detección de outliers) aplicados
a "{resumen.get('dataset', 'un dataset')}", en formato JSON:

{datos_json}

Instrucciones:
- Genera entre 3 y 5 conclusiones claras, en español, para un lector no técnico.
- Cada conclusión debe ser 1-2 oraciones, en formato de lista.
- Traduce los términos técnicos a lenguaje simple (no digas "silhouette score",
  di si los grupos están bien definidos o no; no digas "IQR", di "valores
  fuera de lo común").
- Si hay clusters, describe qué diferencia a cada grupo (usa los promedios).
- Si hay outliers, menciona cuántos son y en qué se diferencian del resto,
  SIN inventar quiénes son.
- Si hay correlaciones relevantes, explica qué relación práctica sugieren,
  dejando claro que es una relación observada y no necesariamente una causa.
- Basa TODO en los datos de arriba. No inventes cifras ni patrones que no
  estén en el JSON.
- No repitas el JSON ni uses jerga estadística en la respuesta.
"""

    return prompt


MODELO_POR_DEFECTO = "llama-3.3-70b-versatile"


def generar_interpretacion(
    resumen: dict,
    api_key: Optional[str] = None,
    modelo: str = MODELO_POR_DEFECTO,
    temperatura: float = 0.4,
    max_tokens: int = 700,
) -> str:

# Se importa Groq solo cuando se necesita, permitiendo que
# las demás funciones sigan funcionando aunque no esté instalado.

    from groq import Groq

    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró la API key de Groq. Configura la variable de "
            "entorno GROQ_API_KEY o pásala como parámetro."
        )

    prompt = construir_prompt(resumen)

    cliente = Groq(api_key=api_key)

    try:

        respuesta = cliente.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente que traduce análisis de datos a lenguaje simple para personas no técnicas.",
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=temperatura,
            max_tokens=max_tokens,
        )

    except Exception as e:

# Maneja posibles errores al conectarse con la API y permite
# mostrar un mensaje de error sin detener el dashboard.

        raise RuntimeError(f"No se pudo generar la interpretación con Groq: {e}") from e

    texto = respuesta.choices[0].message.content

    if not texto or not texto.strip():
        raise RuntimeError("Groq devolvió una respuesta vacía.")

    return texto.strip()


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from modulos.carga_datos import cargar_archivo
    from modulos.procesamiento import procesar
    from modulos.ia import aplicar_ia
    from modulos.analisis_exploratorio import analizar

    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo/ventas_ejemplo.csv"

    carga = cargar_archivo(ruta)

    if not carga.exito:
        print("Error al cargar:", carga.errores)
        sys.exit(1)

    procesamiento = procesar(carga.dataframe)
    resultado_exploratorio = analizar(procesamiento.dataframe, procesamiento.tipos_columna)
    resultado_ia = aplicar_ia(procesamiento.dataframe)

    resumen = construir_resumen(
        resultado_ia=resultado_ia,
        resultado_exploratorio=resultado_exploratorio,
        reporte_limpieza=procesamiento.reporte_limpieza,
        nombre_dataset=ruta,
    )

    print("=== RESUMEN QUE SE LE MANDA AL LLM ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))

    print("\n=== INTERPRETACIÓN GENERADA ===")

    try:
        print(generar_interpretacion(resumen))
    except RuntimeError as e:
        print(f"[AVISO] {e}")
