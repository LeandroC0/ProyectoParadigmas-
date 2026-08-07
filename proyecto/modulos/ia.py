import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


def aplicar_ia(df: pd.DataFrame):

    # ==============================
    # 1. Se detectan las columnas numericas
    # ==============================

    columnas_numericas = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    columnas = []

    for columna in columnas_numericas:

        nombre_columna = columna.lower()

        # Excluir columnas que parecen identificadores
        # Ejemplos:
        # id
        # id_cliente
        # cliente_id
        # codigo
        # codigo_cliente

        es_identificador = (
            nombre_columna == "id"
            or nombre_columna.startswith("id_")
            or nombre_columna.endswith("_id")
            or nombre_columna == "codigo"
            or nombre_columna.startswith("codigo_")
            or nombre_columna.endswith("_codigo")
            or nombre_columna == "código"
            or nombre_columna.startswith("código_")
            or nombre_columna.endswith("_código")
        )

        if not es_identificador:
            columnas.append(columna)

    # ==============================
    # 2. Se validan las columnas
    # ==============================

    if len(columnas) < 2:
        raise ValueError(
            "Se necesitan al menos 2 columnas numéricas "
            "para aplicar clustering y detección de outliers."
        )

    # ==============================
    # 3. Se preparan los datos
    # ==============================

    datos = df[columnas].copy()

    # StandardScaler coloca las variables
    # en una escala comparable

    scaler = StandardScaler()

    datos_escalados = scaler.fit_transform(datos)

    # ==============================
    # 4. Se busca la mejor cantidad de clusters
    # ==============================

    mejor_numero_clusters = 2
    mejor_score = -1

    evaluacion_clusters = []

    # Como máximo probar 5 clusters
    # y nunca más clusters que registros

    max_clusters = min(
        5,
        len(df) - 1
    )

    for numero_clusters in range(
        2,
        max_clusters + 1
    ):

        modelo_prueba = KMeans(
            n_clusters=numero_clusters,
            random_state=42,
            n_init="auto"
        )

        clusters_prueba = (
            modelo_prueba.fit_predict(
                datos_escalados
            )
        )

        score = silhouette_score(
            datos_escalados,
            clusters_prueba
        )

        evaluacion_clusters.append(
            {
                "numero_clusters": numero_clusters,
                "silhouette_score": score
            }
        )

        if score > mejor_score:

            mejor_score = score

            mejor_numero_clusters = (
                numero_clusters
            )

    # ==============================
    # 5. Clustering final y con K-MEANS
    # ==============================

    modelo_kmeans = KMeans(
        n_clusters=mejor_numero_clusters,
        random_state=42,
        n_init="auto"
    )

    clusters = modelo_kmeans.fit_predict(
        datos_escalados
    )

    # ==============================
    # 6. Se hace detección de outliers y con Isolation Forest
    # ==============================

    modelo_outliers = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    outliers = modelo_outliers.fit_predict(
        datos_escalados
    )

    # ==============================
    # 7. Se crea dataframe del resultado
    # ==============================

    resultado = df.copy()

    resultado["cluster"] = clusters

    # Isolation Forest devuelve:

    #  1 = registro normal
    # -1 = outlier

    resultado["outlier"] = [
        "Si" if valor == -1 else "No"
        for valor in outliers
    ]


    agregaciones = {}

    for columna in columnas:

        agregaciones[
            f"{columna}_promedio"
        ] = (
            columna,
            "mean"
        )


    columna_conteo = df.columns[0]

    resumen_clusters = (
        resultado
        .groupby("cluster")
        .agg(
            cantidad_registros=(
                columna_conteo,
                "count"
            ),
            **agregaciones
        )
        .round(2)
        .reset_index()
    )

    # ==============================
    # 8. Se obtienen outliers
    # ==============================

    outliers_detectados = resultado[
        resultado["outlier"] == "Si"
    ].copy()

    cantidad_outliers = len(
        outliers_detectados
    )


    return {

        "dataframe": resultado,

        "columnas_utilizadas": columnas,

        "mejor_numero_clusters":
            mejor_numero_clusters,

        "mejor_silhouette":
            mejor_score,

        "evaluacion_clusters":
            evaluacion_clusters,

        "cantidad_outliers":
            cantidad_outliers,

        "outliers":
            outliers_detectados,

        "resumen_clusters":
            resumen_clusters
    }