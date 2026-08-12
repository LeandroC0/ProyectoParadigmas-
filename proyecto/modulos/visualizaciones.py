

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DISTRIBUCIONES (numéricas y categóricas)
# ============================================================

def graficar_distribucion_numerica(df: pd.DataFrame, columna: str) -> go.Figure:
    """Histograma + boxplot marginal para una columna numérica."""
    fig = px.histogram(
        df,
        x=columna,
        marginal="box",
        title=f"Distribución de {columna}",
        nbins=30,
    )
    fig.update_layout(bargap=0.05)
    return fig


def graficar_top_categorias(estadisticas_categoricas: dict, columna: str) -> go.Figure:
    """Barras con las categorías más frecuentes de una columna categórica."""
    stats = estadisticas_categoricas[columna]
    top = stats.top_categorias if hasattr(stats, "top_categorias") else stats["top_categorias"]

    fig = px.bar(
        x=list(top.keys()),
        y=list(top.values()),
        labels={"x": columna, "y": "Frecuencia"},
        title=f"Categorías más frecuentes en {columna}",
    )
    return fig


# ============================================================
# 2. CORRELACIONES
# ============================================================

def graficar_matriz_correlacion(matriz_correlacion: Optional[pd.DataFrame]) -> Optional[go.Figure]:
    """Mapa de calor de la matriz de correlación. Devuelve None si no hay matriz."""
    if matriz_correlacion is None or matriz_correlacion.empty:
        return None

    fig = px.imshow(
        matriz_correlacion,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlación",
    )
    return fig


def graficar_dispersion_correlacion(df: pd.DataFrame, columna_a: str, columna_b: str) -> go.Figure:
    """Diagrama de dispersión para un par de columnas correlacionadas."""
    fig = px.scatter(
        df,
        x=columna_a,
        y=columna_b,
        title=f"{columna_a} vs {columna_b}",
    )
    return fig


# ============================================================
# 3. CLUSTERING Y OUTLIERS (salida de ia.aplicar_ia)
# ============================================================

def graficar_clusters(resultado_ia: dict) -> go.Figure:
    """
    Proyecta las columnas usadas para clustering en 2D (PCA si son más de 2)
    y colorea por cluster. Los outliers (columna 'outlier': 'Si'/'No') se
    marcan con un símbolo distinto.
    """
    df = resultado_ia["dataframe"]
    columnas = resultado_ia["columnas_utilizadas"]

    datos_escalados = StandardScaler().fit_transform(df[columnas])

    if len(columnas) > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(datos_escalados)
        eje_x, eje_y = "Componente 1", "Componente 2"
    else:
        coords = datos_escalados
        eje_x, eje_y = f"{columnas[0]} (estandarizado)", f"{columnas[1]} (estandarizado)"

    datos_plot = pd.DataFrame(coords, columns=[eje_x, eje_y])
    datos_plot["cluster"] = df["cluster"].astype(str).values
    datos_plot["outlier"] = df["outlier"].values

    fig = px.scatter(
        datos_plot,
        x=eje_x,
        y=eje_y,
        color="cluster",
        symbol="outlier",
        symbol_map={"No": "circle", "Si": "x"},
        title=(
            f"Clusters detectados (k={resultado_ia['mejor_numero_clusters']}, "
            f"silhouette={resultado_ia['mejor_silhouette']:.2f})"
        ),
    )
    return fig


def graficar_resumen_clusters(resultado_ia: dict) -> go.Figure:
    """
    Barras comparando el promedio de cada variable por cluster, en paneles
    separados (uno por variable, cada uno con su propia escala) por la misma
    razón que outliers_por_variable: las variables tienen unidades muy distintas.
    """
    resumen = resultado_ia["resumen_clusters"].copy()
    columnas_promedio = [c for c in resumen.columns if c.endswith("_promedio")]

    resumen_largo = resumen.melt(
        id_vars=["cluster", "cantidad_registros"],
        value_vars=columnas_promedio,
        var_name="variable",
        value_name="promedio",
    )
    resumen_largo["variable"] = resumen_largo["variable"].str.replace("_promedio", "", regex=False)
    resumen_largo["cluster"] = resumen_largo["cluster"].astype(str)

    fig = px.bar(
        resumen_largo,
        x="cluster",
        y="promedio",
        color="cluster",
        facet_col="variable",
        facet_col_wrap=min(len(columnas_promedio), 4),
        title="Promedio de cada variable por cluster",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def graficar_evaluacion_clusters(resultado_ia: dict) -> go.Figure:
    """Línea del silhouette score probado para cada k (justifica el número de clusters elegido)."""
    evaluacion = pd.DataFrame(resultado_ia["evaluacion_clusters"])

    fig = px.line(
        evaluacion,
        x="numero_clusters",
        y="silhouette_score",
        markers=True,
        title="Silhouette score según número de clusters",
    )
    fig.add_vline(
        x=resultado_ia["mejor_numero_clusters"],
        line_dash="dash",
        annotation_text="k elegido",
    )
    return fig


def graficar_outliers_por_variable(resultado_ia: dict) -> go.Figure:
    """
    Boxplots de las variables usadas en clustering, uno por variable (paneles
    separados, cada uno con su propia escala) para poder comparar los outliers
    aunque las variables tengan unidades muy distintas (ej. edad vs ingresos).
    """
    df = resultado_ia["dataframe"]
    columnas = resultado_ia["columnas_utilizadas"]

    datos_largo = df.melt(
        id_vars=["outlier"],
        value_vars=columnas,
        var_name="variable",
        value_name="valor",
    )

    fig = px.box(
        datos_largo,
        x="outlier",
        y="valor",
        color="outlier",
        facet_col="variable",
        facet_col_wrap=min(len(columnas), 4),
        title="Distribución por variable (outliers resaltados)",
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


# ============================================================
# 4. ORQUESTADOR: genera todo lo disponible según los datos recibidos
# ============================================================

def _es_columna_identificador(nombre_columna: str) -> bool:
    """
    Detecta columnas tipo id_cliente, codigo, etc. Misma regla que usa
    ia.py para excluirlas del clustering; aquí se usa para no graficar
    su 'distribución', que no tiene significado real.
    """
    nombre = nombre_columna.lower()
    return (
        nombre == "id"
        or nombre.startswith("id_")
        or nombre.endswith("_id")
        or nombre == "codigo"
        or nombre.startswith("codigo_")
        or nombre.endswith("_codigo")
        or nombre == "código"
        or nombre.startswith("código_")
        or nombre.endswith("_código")
    )


def generar_todas_visualizaciones(
    df: pd.DataFrame,
    resultado_exploratorio,
    resultado_ia: Optional[dict] = None,
    max_distribuciones: int = 6,
    max_dispersion: int = 3,
    umbral_unicidad_categorica: float = 0.8,
) -> dict[str, go.Figure]:
    """
    Genera un diccionario {nombre_grafico: figura_plotly} con todo lo que se
    pueda construir a partir de las salidas de procesamiento, analisis_exploratorio
    e ia. Pensado para iterarse directo en Streamlit:

        for nombre, fig in figuras.items():
            st.plotly_chart(fig, use_container_width=True)

    Excluye automáticamente columnas identificadoras (id_cliente, codigo, etc.)
    y columnas categóricas casi únicas (ej. nombres de personas), que no
    aportan una visualización con significado real.
    """
    figuras: dict[str, go.Figure] = {}

    # --- Distribuciones numéricas (se excluyen identificadores) ---
    columnas_numericas = [
        c for c in resultado_exploratorio.estadisticas_numericas.keys()
        if not _es_columna_identificador(c)
    ]
    for columna in columnas_numericas[:max_distribuciones]:
        figuras[f"distribucion_{columna}"] = graficar_distribucion_numerica(df, columna)

    # --- Top categorías por columna categórica (se excluyen las casi únicas) ---
    for columna, stats in resultado_exploratorio.estadisticas_categoricas.items():
        proporcion_unicos = stats.valores_unicos / stats.conteo if stats.conteo else 0
        if proporcion_unicos >= umbral_unicidad_categorica:
            continue
        figuras[f"categorias_{columna}"] = graficar_top_categorias(
            resultado_exploratorio.estadisticas_categoricas, columna
        )

    # --- Matriz de correlación ---
    fig_corr = graficar_matriz_correlacion(resultado_exploratorio.matriz_correlacion)
    if fig_corr is not None:
        figuras["matriz_correlacion"] = fig_corr

    # --- Dispersión para las correlaciones más relevantes ---
    for par in resultado_exploratorio.correlaciones_relevantes[:max_dispersion]:
        nombre = f"dispersion_{par.columna_a}_vs_{par.columna_b}"
        figuras[nombre] = graficar_dispersion_correlacion(df, par.columna_a, par.columna_b)

    # --- Clustering y outliers (si se pasó el resultado de ia.py) ---
    if resultado_ia is not None:
        figuras["clusters"] = graficar_clusters(resultado_ia)
        figuras["resumen_clusters"] = graficar_resumen_clusters(resultado_ia)
        figuras["evaluacion_clusters"] = graficar_evaluacion_clusters(resultado_ia)
        figuras["outliers_por_variable"] = graficar_outliers_por_variable(resultado_ia)

    return figuras


# ============================================================
# PRUEBA DIRECTA POR TERMINAL
# (misma convención que los demás módulos: py -m modulos.visualizaciones <ruta_csv>)
# ============================================================

if __name__ == "__main__":
    import sys
    from modulos.carga_datos import cargar_archivo
    from modulos.procesamiento import procesar
    from modulos.analisis_exploratorio import analizar
    from modulos.ia import aplicar_ia

    ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo/ventas_ejemplo.csv"

    carga = cargar_archivo(ruta)
    if not carga.exito:
        print("Error al cargar:", carga.errores)
        sys.exit(1)

    procesado = procesar(carga.dataframe)
    exploratorio = analizar(procesado.dataframe, procesado.tipos_columna)
    resultado_ia = aplicar_ia(procesado.dataframe)

    figuras = generar_todas_visualizaciones(
        df=procesado.dataframe,
        resultado_exploratorio=exploratorio,
        resultado_ia=resultado_ia,
    )

    print(f"Se generaron {len(figuras)} visualizaciones:")
    for nombre in figuras:
        print(f"  - {nombre}")

    # Guarda todas como HTML para revisarlas sin tener que llamar a .show() una por una
    import os
    os.makedirs("visualizaciones_html", exist_ok=True)
    for nombre, fig in figuras.items():
        fig.write_html(f"visualizaciones_html/{nombre}.html")
    print(f"\nSe guardaron en la carpeta 'visualizaciones_html/'")
