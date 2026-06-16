# TFM — Predicción de gravedad y mapa de riesgo de siniestros viales en Bogotá

Trabajo Fin de Máster · Máster en Data Science, Big Data & Business Analytics (UCM)

**Autor:** Ovidio Almanza Ledesma

## Resumen

Proyecto de análisis de un dataset público (orientación Data Scientist). A partir del histórico de siniestros viales georreferenciados de Bogotá se construye un modelo de clasificación de la **gravedad** del siniestro y una **productivización** en forma de mapa de riesgo, para apoyar la priorización de intervenciones de seguridad vial (marco *Visión Cero*).

**Pregunta de negocio:** ¿dónde, cuándo y bajo qué condiciones es más probable que ocurra un siniestro grave, para priorizar intervenciones con presupuesto limitado?

## Datos

- **Fuente:** *Histórico Siniestros Bogotá D.C.* — Secretaría Distrital de Movilidad (Datos Abiertos Bogotá).
- **Volumen:** 209.861 siniestros georreferenciados (2015–2021).
- **Licencia de los datos:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Acceso:** API Esri REST, con descarga reproducible desde el propio notebook.
- Los datos **no se versionan** en el repositorio (ver `.gitignore`); se descargan automáticamente al ejecutar el notebook 01.

## Estructura del repositorio

```
notebooks/
  01_eda_y_validacion.ipynb   # Carga reproducible, EDA y diseño de validación
app/                          # Productivización (Streamlit) — en desarrollo
data/                         # Datos (ignorado por git)
requirements.txt
README.md
LICENSE
```

## Reproducibilidad

1. (Opcional) crear un entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar los notebooks en orden (`01`, `02`, …). El notebook 01 descarga los datos automáticamente vía API y deja un artefacto limpio en `data/processed/`.
3. También se pueden abrir directamente en Google Colab.

## Metodología (resumen)

Análisis exploratorio y control de calidad de datos · saneamiento defensivo de tipos · enriquecimiento con tablas relacionales · ingeniería de variables espacio-temporales · modelos con justificación de métrica y estrategia de validación temporal · interpretabilidad (SHAP) · mapa de riesgo como productivización.

## Licencia

Código bajo licencia **MIT** (ver `LICENSE`). Los datos pertenecen a la Secretaría Distrital de Movilidad de Bogotá y se usan bajo licencia **CC BY 4.0**.
