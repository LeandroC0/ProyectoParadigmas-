# Proyecto: Análisis Automatizado de Datos

## Módulos entregados (Carga de Datos + Procesamiento)



### Instalación

```bash
pip install -r requirements.txt
```

### 1. `modulos/carga_datos.py`

Importa y valida archivos CSV/Excel.



### 2. `modulos/procesamiento.py`

Detecta el tipo de cada columna (numérica, categórica, fecha, booleana) y
limpia los datos (nulos, duplicados, tipos mal escritos).


### Probar sin Streamlit

Cada módulo se puede correr directo desde consola para probarlo:

```bash
python modulos/carga_datos.py datos_ejemplo/ventas_ejemplo.csv
python modulos/procesamiento.py datos_ejemplo/ventas_ejemplo.csv
py analisis_exploratorio.py ../datos_ejemplo/ventas_ejemplo.csv
```



## Pendiente 

- `analisis_exploratorio.py` — estadísticas descriptivas, distribuciones, correlaciones
- `ia.py` — clustering (K-Means/DBSCAN), detección de outliers
- `interpretacion.py` — conclusiones en lenguaje natural (LLM vía Groq)
- `visualizaciones.py` — histogramas, boxplots, dispersión, heatmaps
- `reporte.py` — exportar PDF/HTML
- `app.py` — integración final en Streamlit


