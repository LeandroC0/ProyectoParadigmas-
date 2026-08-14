
import streamlit as st

from modulos.carga_datos import cargar_archivo
from modulos.procesamiento import procesar
from modulos.analisis_exploratorio import analizar
from modulos.ia import aplicar_ia
from modulos.visualizaciones import generar_todas_visualizaciones
from modulos.interpretacion_llm import construir_resumen, generar_interpretacion


st.set_page_config(page_title="Analizador de Datos", layout="wide")
st.title("Sistema de Análisis Automatizado de Datos")

archivo = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])


if archivo is None:
    st.info("Sube un archivo para comenzar el análisis.")
    st.stop()


carga = cargar_archivo(archivo, nombre_archivo=archivo.name)

if not carga.exito:
    st.error("No se pudo procesar el archivo:")
    for error in carga.errores:
        st.write(f"- {error}")
    st.stop()

for advertencia in carga.advertencias:
    st.warning(advertencia)

st.success(f"Archivo cargado: {carga.filas} filas x {carga.columnas} columnas")

# --- 2. Procesamiento ---
@st.cache_data(show_spinner="Procesando datos...")
def _pipeline_completo(_df, nombre_archivo):
    procesamiento = procesar(_df)
    exploratorio = analizar(procesamiento.dataframe, procesamiento.tipos_columna)
    resultado_ia = aplicar_ia(procesamiento.dataframe)
    figuras = generar_todas_visualizaciones(
        df=procesamiento.dataframe,
        resultado_exploratorio=exploratorio,
        resultado_ia=resultado_ia,
    )
    return procesamiento, exploratorio, resultado_ia, figuras


try:
    procesamiento, exploratorio, resultado_ia, figuras = _pipeline_completo(
        carga.dataframe, archivo.name
    )
except ValueError as e:
    # ia.py lanza ValueError si no hay suficientes columnas numéricas
    st.error(f"No se pudo aplicar el módulo de IA: {e}")
    st.stop()

# --- 3. Pestañas de resultados ---
tab_datos, tab_exploratorio, tab_ia, tab_graficos, tab_interpretacion = st.tabs(
    ["Datos", "Análisis exploratorio", "IA (clusters y outliers)", "Gráficos", "Interpretación"]
)

with tab_datos:
    st.subheader("Vista previa de los datos limpios")
    st.dataframe(procesamiento.dataframe.head(20))

    st.subheader("Tipos de columna detectados")
    st.json(procesamiento.tipos_columna)

    st.subheader("Reporte de limpieza")
    st.json(procesamiento.reporte_limpieza)

with tab_exploratorio:
    st.subheader("Resumen general")
    st.json(exploratorio.resumen_general)

    st.subheader("Estadísticas numéricas")
    for columna, stats in exploratorio.estadisticas_numericas.items():
        with st.expander(columna):
            st.write(
                f"Media: {stats.media:.2f} | Mediana: {stats.mediana:.2f} | "
                f"Desv. estándar: {stats.desviacion_estandar:.2f}"
            )
            st.write(
                f"Mínimo: {stats.minimo:.2f} | Máximo: {stats.maximo:.2f} | "
                f"Outliers (IQR): {stats.valores_atipicos_iqr}"
            )
            st.write(f"Distribución: {stats.interpretacion_distribucion}")

    if exploratorio.correlaciones_relevantes:
        st.subheader("Correlaciones relevantes")
        for par in exploratorio.correlaciones_relevantes:
            st.write(
                f"**{par.columna_a} ↔ {par.columna_b}**: {par.coeficiente} "
                f"({par.fuerza}, {par.direccion})"
            )

    for advertencia in exploratorio.advertencias:
        st.warning(advertencia)

with tab_ia:
    st.metric("Número de clusters óptimo", resultado_ia["mejor_numero_clusters"])
    st.metric("Silhouette score", round(resultado_ia["mejor_silhouette"], 3))
    st.metric("Outliers detectados", resultado_ia["cantidad_outliers"])

    st.subheader("Resumen por cluster")
    st.dataframe(resultado_ia["resumen_clusters"])

    if resultado_ia["cantidad_outliers"] > 0:
        st.subheader("Registros marcados como outlier")
        st.dataframe(resultado_ia["outliers"])

with tab_graficos:
    # Todas las figuras vienen ya listas desde visualizaciones.py
    for nombre, figura in figuras.items():
        st.plotly_chart(figura, use_container_width=True)

with tab_interpretacion:
    st.subheader("Conclusiones generadas por IA (LLM)")

    if st.button("Generar interpretación"):
        resumen = construir_resumen(
            resultado_ia=resultado_ia,
            resultado_exploratorio=exploratorio,
            reporte_limpieza=procesamiento.reporte_limpieza,
            nombre_dataset=archivo.name,
        )
        try:
            with st.spinner("Consultando al modelo..."):
                texto = generar_interpretacion(resumen)
            st.markdown(texto)
        except RuntimeError as e:
            st.error(str(e))
    else:
        st.caption(
        )