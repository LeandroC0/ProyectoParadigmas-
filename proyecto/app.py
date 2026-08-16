import io

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)

from modulos.carga_datos import cargar_archivo
from modulos.procesamiento import procesar
from modulos.analisis_exploratorio import analizar
from modulos.ia import aplicar_ia
from modulos.visualizaciones import generar_todas_visualizaciones
from modulos.interpretacion_llm import construir_resumen, generar_interpretacion

# Colores del reporte, para no repetir el mismo hex por todo el archivo
COLOR_PRINCIPAL = colors.HexColor("#2C3E50")
COLOR_ACENTO = colors.HexColor("#2E86AB")
COLOR_FILA_PAR = colors.HexColor("#F4F6F7")


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
            st.session_state["texto_interpretacion"] = texto
        except RuntimeError as e:
            st.error(str(e))

    if "texto_interpretacion" in st.session_state:
        st.markdown(st.session_state["texto_interpretacion"])
    else:
        st.caption("Presiona el botón para generar las conclusiones con IA.")


# --- 4. Reporte final en PDF ---

# Estilos propios, basados en los estilos por defecto de reportlab
# pero con la paleta de colores del reporte

def _construir_estilos():
    base = getSampleStyleSheet()

    base.add(ParagraphStyle(
        name="TituloReporte",
        parent=base["Title"],
        textColor=COLOR_PRINCIPAL,
        fontSize=24,
        spaceAfter=6,
    ))

    base.add(ParagraphStyle(
        name="Subtitulo",
        parent=base["Normal"],
        textColor=colors.grey,
        fontSize=11,
        spaceAfter=18,
    ))

    base.add(ParagraphStyle(
        name="SeccionTitulo",
        parent=base["Heading2"],
        textColor=COLOR_ACENTO,
        fontSize=14,
        spaceBefore=14,
        spaceAfter=8,
    ))

    base.add(ParagraphStyle(
        name="TextoNormal",
        parent=base["Normal"],
        fontSize=10,
        leading=14,
    ))

    return base


def _tabla_con_estilo(filas):
    tabla = Table(filas, hAlign="LEFT")

    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRINCIPAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D8DC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_FILA_PAR]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    tabla.setStyle(TableStyle(estilo))
    return tabla


def _pie_de_pagina(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.grey)
    canvas_obj.drawString(2 * cm, 1.2 * cm, "Generado automáticamente por el sistema de análisis")
    canvas_obj.drawRightString(19 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas_obj.restoreState()


def _generar_pdf_reporte(nombre_archivo, procesamiento, exploratorio, resultado_ia, texto_interpretacion):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    estilos = _construir_estilos()
    contenido = []

    # Portada del reporte
    contenido.append(Paragraph("Reporte de Análisis de Datos", estilos["TituloReporte"]))
    contenido.append(Paragraph(f"Archivo analizado: {nombre_archivo}", estilos["Subtitulo"]))
    contenido.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACENTO))
    contenido.append(Spacer(1, 16))

    # Reporte de limpieza
    contenido.append(Paragraph("Reporte de limpieza", estilos["SeccionTitulo"]))
    for clave, valor in procesamiento.reporte_limpieza.items():
        contenido.append(Paragraph(f"<b>{clave}:</b> {valor}", estilos["TextoNormal"]))
    contenido.append(Spacer(1, 12))

    # Estadísticas numéricas
    contenido.append(Paragraph("Estadísticas numéricas", estilos["SeccionTitulo"]))
    filas_tabla = [["Columna", "Media", "Mediana", "Desv. estándar", "Outliers (IQR)"]]
    for columna, stats in exploratorio.estadisticas_numericas.items():
        filas_tabla.append([
            columna,
            f"{stats.media:.2f}",
            f"{stats.mediana:.2f}",
            f"{stats.desviacion_estandar:.2f}",
            str(stats.valores_atipicos_iqr),
        ])
    contenido.append(_tabla_con_estilo(filas_tabla))
    contenido.append(Spacer(1, 12))

    # Correlaciones relevantes
    if exploratorio.correlaciones_relevantes:
        contenido.append(Paragraph("Correlaciones relevantes", estilos["SeccionTitulo"]))
        for par in exploratorio.correlaciones_relevantes:
            contenido.append(Paragraph(
                f"<b>{par.columna_a} - {par.columna_b}:</b> {par.coeficiente} "
                f"({par.fuerza}, {par.direccion})",
                estilos["TextoNormal"],
            ))
        contenido.append(Spacer(1, 12))

    contenido.append(PageBreak())

    # Clustering y outliers
    contenido.append(Paragraph("IA: clustering y outliers", estilos["SeccionTitulo"]))
    contenido.append(Paragraph(
        f"<b>Número de clusters óptimo:</b> {resultado_ia['mejor_numero_clusters']}", estilos["TextoNormal"]
    ))
    contenido.append(Paragraph(
        f"<b>Silhouette score:</b> {round(resultado_ia['mejor_silhouette'], 3)}", estilos["TextoNormal"]
    ))
    contenido.append(Paragraph(
        f"<b>Outliers detectados:</b> {resultado_ia['cantidad_outliers']}", estilos["TextoNormal"]
    ))
    contenido.append(Spacer(1, 12))

    filas_clusters = [list(resultado_ia["resumen_clusters"].columns)]
    filas_clusters += resultado_ia["resumen_clusters"].astype(str).values.tolist()
    contenido.append(_tabla_con_estilo(filas_clusters))
    contenido.append(Spacer(1, 16))

    # Interpretación del LLM
    contenido.append(Paragraph("Interpretación (IA generativa)", estilos["SeccionTitulo"]))
    if texto_interpretacion:
        for parrafo in texto_interpretacion.split("\n"):
            if parrafo.strip():
                contenido.append(Paragraph(parrafo, estilos["TextoNormal"]))
    else:
        contenido.append(Paragraph(
            "No se generó interpretación con IA para este reporte.", estilos["TextoNormal"]
        ))

    doc.build(contenido, onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)
    buffer.seek(0)
    return buffer


st.divider()
st.subheader("Descargar informe")

pdf_buffer = _generar_pdf_reporte(
    nombre_archivo=archivo.name,
    procesamiento=procesamiento,
    exploratorio=exploratorio,
    resultado_ia=resultado_ia,
    texto_interpretacion=st.session_state.get("texto_interpretacion"),
)

st.download_button(
    label="Descargar informe en PDF",
    data=pdf_buffer,
    file_name=f"reporte_{archivo.name}.pdf",
    mime="application/pdf",
)