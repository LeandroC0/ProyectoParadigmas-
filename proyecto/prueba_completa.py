from modulos.carga_datos import cargar_archivo
from modulos.procesamiento import procesar
from modulos.ia import aplicar_ia


# ==============================
# 1. Se carga el archivo CSV
# ==============================

ruta = "datos_ejemplo/ventas_ejemplo.csv"

carga = cargar_archivo(ruta)

# Se valida que el archivo se haya cargado correctamente

if not carga.exito:
    print("Error al cargar el archivo:")
    print(carga.errores)
    exit()

print("\n==============================")
print("1. ARCHIVO CARGADO")
print("==============================")

print("Filas:", carga.filas)
print("Columnas:", carga.columnas)


# ==============================
# 2. Se procesan y limpian los datos
# ==============================

procesamiento = procesar(carga.dataframe)

# Se guarda el dataframe ya limpio para utilizarlo en la IA

df_limpio = procesamiento.dataframe


# Se muestran los tipos de columnas detectados

print("\n==============================")
print("2. TIPOS DE COLUMNAS")
print("==============================")

for columna, tipo in procesamiento.tipos_columna.items():
    print(columna, "->", tipo)


# Se muestra el resultado del proceso de limpieza

print("\n==============================")
print("3. REPORTE DE LIMPIEZA")
print("==============================")

for clave, valor in procesamiento.reporte_limpieza.items():
    print(clave, ":", valor)


# ==============================
# 3. Se aplica el módulo de IA
# ==============================

resultado_ia = aplicar_ia(df_limpio)


# ==============================
# 4. Se evalúan las diferentes cantidades de clusters
# ==============================

print("\n==============================")
print("4. EVALUACIÓN DE CLUSTERS")
print("==============================")

# Se muestran los resultados obtenidos al probar
# diferentes cantidades de clusters

for evaluacion in resultado_ia["evaluacion_clusters"]:

    numero = evaluacion["numero_clusters"]
    score = evaluacion["silhouette_score"]

    print(
        f"{numero} clusters -> "
        f"Silhouette Score: {score:.4f}"
    )


# Se muestra cuál cantidad de clusters obtuvo el mejor resultado

print(
    "\nMejor cantidad de clusters:",
    resultado_ia["mejor_numero_clusters"]
)

print(
    "Mejor Silhouette Score:",
    round(resultado_ia["mejor_silhouette"], 4)
)


# ==============================
# 5. Se muestran los resultados de la IA
# ==============================

print("\n==============================")
print("5. RESULTADO DE IA")
print("==============================")

df_resultado = resultado_ia["dataframe"]

# Se muestran los datos originales junto con
# el cluster y el resultado de outlier

print(
    df_resultado[
        [
            "id_cliente",
            "nombre",
            "region",
            "edad",
            "ingresos_mensuales",
            "gasto_publicidad",
            "ventas",
            "cluster",
            "outlier"
        ]
    ].to_string(index=False)
)


# ==============================
# 6. Se muestran los outliers detectados
# ==============================

print("\n==============================")
print("6. OUTLIERS DETECTADOS")
print("==============================")

print(
    "Cantidad de outliers:",
    resultado_ia["cantidad_outliers"]
)

outliers = resultado_ia["outliers"]

# Si no existen outliers se muestra un mensaje
# y en caso contrario se muestran los registros encontrados

if outliers.empty:

    print("No se detectaron outliers.")

else:

    print(
        outliers[
            [
                "id_cliente",
                "nombre",
                "region",
                "edad",
                "ingresos_mensuales",
                "gasto_publicidad",
                "ventas",
                "cluster"
            ]
        ].to_string(index=False)
    )


# ==============================
# 7. Se muestra el resumen de los clusters
# ==============================

print("\n==============================")
print("7. RESUMEN DE CLUSTERS")
print("==============================")

# Se muestran los promedios y la cantidad
# de registros encontrados en cada cluster

print(
    resultado_ia["resumen_clusters"].to_string(
        index=False
    )
)


# ==============================
# 8. Se muestran las variables utilizadas por la IA
# ==============================

print("\n==============================")
print("8. VARIABLES UTILIZADAS POR LA IA")
print("==============================")

# Se muestran las columnas numéricas que
# fueron utilizadas para realizar el análisis

for columna in resultado_ia["columnas_utilizadas"]:
    print("-", columna)