"""
App Streamlit — Riesgo de siniestros viales en Bogotá
TFM · Ovidio Almanza Ledesma

Ejecutar en local:   streamlit run app.py
Artefactos necesarios (generados desde el notebook):
  - catboost_gravedad.cbm
  - app_zonas.csv
  - app_opciones.json
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
import branca
import h3
from streamlit_folium import st_folium
from catboost import CatBoostClassifier

st.set_page_config(page_title="Riesgo vial Bogotá", page_icon="🚦", layout="wide")

BASE = Path(__file__).resolve().parent  # carpeta donde está app.py

CLASES = ["Solo daños", "Con heridos", "Con muertos"]
_boundary = getattr(h3, "cell_to_boundary", None) or h3.h3_to_geo_boundary


# ------------------ Carga de artefactos (cacheada) ------------------
@st.cache_resource
def cargar_modelo():
    m = CatBoostClassifier()
    m.load_model(str(BASE / "catboost_gravedad.cbm"))
    return m

@st.cache_data
def cargar_zonas():
    return pd.read_csv(BASE / "app_zonas.csv")

@st.cache_data
def cargar_opciones():
    with open(BASE / "app_opciones.json", encoding="utf-8") as f:
        return json.load(f)

try:
    MODELO = cargar_modelo()
    ZONAS = cargar_zonas()
    OPC = cargar_opciones()
except Exception as e:
    st.error("No se encontraron los artefactos (catboost_gravedad.cbm, app_zonas.csv, "
             "app_opciones.json). Genéralos desde el notebook y colócalos junto a app.py.")
    st.stop()


# ------------------ Cabecera ------------------
st.title("🚦 Riesgo de siniestros viales en Bogotá")
st.caption("TFM · Predicción de gravedad y mapa de riesgo · datos 2015–2021 (CC BY 4.0)")

tab_mapa, tab_sim, tab_info = st.tabs(["️ Mapa de riesgo", " Simulador de riesgo", "ℹ️ Acerca de"])


# ------------------ Tab 1: Mapa ------------------
with tab_mapa:
    st.subheader("Tasa de fatalidad observada por zona")
    c1, c2 = st.columns([1, 3])
    with c1:
        metrica = st.radio("Colorear por", ["Tasa de fatalidad", "Nº de fatales", "Volumen"], index=0)
        nmax = max(int(ZONAS["n"].quantile(0.90)), 100)
        min_n = st.slider("Mínimo de siniestros por zona (mapa)", 0, nmax, 50, step=10,
                          help="Oculta en el mapa las zonas con pocos siniestros, donde la tasa es menos fiable.")
        col_tab = {"Tasa de fatalidad": "tasa_fatal", "Nº de fatales": "n_fatal", "Volumen": "n"}[metrica]
        etiqueta = {"Tasa de fatalidad": "tasa de fatalidad", "Nº de fatales": "nº de fatales", "Volumen": "volumen"}[metrica]
        st.markdown(f"**Localidades por {etiqueta}:**")
        agg = (ZONAS.groupby("localidad", as_index=False)
               .agg(n=("n", "sum"), n_fatal=("n_fatal", "sum")))
        agg["tasa_fatal"] = (agg["n_fatal"] / agg["n"] * 100).round(2)
        top = agg.sort_values(col_tab, ascending=False).head(8)
        st.dataframe(top[["localidad", "n", "n_fatal", "tasa_fatal"]]
                     .rename(columns={"localidad": "Localidad", "n": "Siniestros",
                                      "n_fatal": "Fatales", "tasa_fatal": "Tasa %"}),
                     hide_index=True, use_container_width=True)
    with c2:
        z = ZONAS[ZONAS["n"] >= min_n].copy()
        col_map = {"Tasa de fatalidad": "tasa_fatal", "Nº de fatales": "n_fatal", "Volumen": "n"}
        col = col_map[metrica]
        vmax = float(z[col].quantile(0.95)) if len(z) else 1.0
        cmap = branca.colormap.LinearColormap(["#2c7fb8", "#ffffb2", "#f03b20"],
                                              vmin=float(z[col].min()) if len(z) else 0, vmax=vmax,
                                              caption=metrica)
        m = folium.Map(location=[4.65, -74.10], zoom_start=11, tiles="cartodbpositron")
        for _, r in z.iterrows():
            try:
                pts = [[lat, lon] for lat, lon in _boundary(r["h3_res7"])]
            except Exception:
                continue
            folium.Polygon(pts, weight=0, fill=True,
                           fill_color=cmap(min(r[col], vmax)), fill_opacity=0.6,
                           popup=folium.Popup(f"<b>{r['localidad']}</b><br>Siniestros: {int(r['n'])}<br>"
                                              f"Fatales: {int(r['n_fatal'])}<br>Tasa: {r['tasa_fatal']}%",
                                              max_width=220)).add_to(m)
        cmap.add_to(m)
        st_folium(m, height=520, use_container_width=True, returned_objects=[])
    st.info("La letalidad NO se concentra donde hay más siniestros: el núcleo urbano (azul) tiene "
            "mucho volumen pero baja tasa, mientras la periferia sur se ilumina en rojo.")


# ------------------ Tab 2: Simulador ------------------
def construir_fila(clase, localidad, causa, hora, finde, ciclista, peaton, pasajero, moto, bici, bus, camion, n_veh):
    fila = dict(OPC["defaults"])
    fila["CLASE_ACC_norm"] = clase
    fila["LOCALIDAD_norm"] = localidad
    fila["grupo_causa"] = causa
    if "hora" in fila: fila["hora"] = float(hora)
    if "hora_sin" in fila: fila["hora_sin"] = math.sin(2 * math.pi * hora / 24)
    if "hora_cos" in fila: fila["hora_cos"] = math.cos(2 * math.pi * hora / 24)
    franja = "Madrugada" if hora < 6 else "Mañana" if hora < 12 else "Tarde" if hora < 18 else "Noche"
    if "franja_horaria" in fila: fila["franja_horaria"] = franja
    if "es_fin_semana" in fila: fila["es_fin_semana"] = bool(finde)
    if "es_finde_o_festivo" in fila: fila["es_finde_o_festivo"] = bool(finde)
    flags = {"involucra_ciclista": ciclista, "involucra_peaton": peaton, "involucra_pasajero": pasajero,
             "involucra_moto": moto, "involucra_motociclista": moto, "involucra_bici": bici,
             "involucra_bus": bus, "involucra_camion": camion}
    for k, v in flags.items():
        if k in fila: fila[k] = bool(v)
    if "n_vehiculos" in fila: fila["n_vehiculos"] = float(n_veh)
    noche = franja in ("Noche", "Madrugada")
    if "moto_noche" in fila: fila["moto_noche"] = bool(moto) and noche
    if "bici_finde" in fila: fila["bici_finde"] = bool(bici) and bool(finde)
    return fila


def predecir(fila):
    X = pd.DataFrame([[fila[c] for c in OPC["feature_cols"]]], columns=OPC["feature_cols"])
    for c in OPC["cat_cols"]:
        X[c] = X[c].astype("string").fillna("SIN_DATO")
    for c in OPC["num_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in OPC["bool_cols"]:
        X[c] = X[c].astype(bool)
    proba = MODELO.predict_proba(X)[0]
    return {CLASES[i]: float(proba[i]) for i in range(3)}


with tab_sim:
    st.subheader("Simulador: ¿qué gravedad predice el modelo?")
    st.caption("Define las condiciones del siniestro y el modelo CatBoost estima la probabilidad de cada gravedad.")
    a, b, c = st.columns(3)
    with a:
        clase = st.selectbox("Tipo de siniestro", OPC["clase"],
                             index=OPC["clase"].index("ATROPELLO") if "ATROPELLO" in OPC["clase"] else 0)
        localidad = st.selectbox("Localidad", OPC["localidad"])
        causa = st.selectbox("Causa principal", OPC["causa"])
    with b:
        hora = st.slider("Hora del día", 0, 23, 22)
        n_veh = st.slider("Nº de vehículos", 1, 6, 1)
        finde = st.checkbox("Fin de semana")
    with c:
        ciclista = st.checkbox("Involucra ciclista")
        peaton = st.checkbox("Involucra peatón", value=True)
        pasajero = st.checkbox("Involucra pasajero")
        moto = st.checkbox("Involucra moto")
        bus = st.checkbox("Involucra bus")
        camion = st.checkbox("Involucra camión")
    bici = ciclista

    fila = construir_fila(clase, localidad, causa, hora, finde, ciclista, peaton, pasajero, moto, bici, bus, camion, n_veh)
    pred = predecir(fila)
    p_fatal = pred["Con muertos"]

    st.markdown("### Resultado")
    m1, m2, m3 = st.columns(3)
    m1.metric("Solo daños", f"{pred['Solo daños']*100:.1f}%")
    m2.metric("Con heridos", f"{pred['Con heridos']*100:.1f}%")
    m3.metric("Con muertos", f"{p_fatal*100:.1f}%")
    st.bar_chart(pd.DataFrame({"Probabilidad": pred}))

    base_fatal = 0.016
    factor = p_fatal / base_fatal if base_fatal else 0
    if p_fatal >= 0.10:
        st.error(f"⚠️ Riesgo de fatalidad ALTO: {p_fatal*100:.1f}% (≈ {factor:.0f}× la media de la ciudad).")
    elif p_fatal >= 0.04:
        st.warning(f"Riesgo de fatalidad MODERADO: {p_fatal*100:.1f}% (≈ {factor:.0f}× la media).")
    else:
        st.success(f"Riesgo de fatalidad BAJO: {p_fatal*100:.1f}%.")
    st.caption("La media de fatalidad de la ciudad es ~1,6%. El modelo prioriza recuperar casos fatales, "
               "por lo que tiende a sobre-señalar contextos de riesgo (alta sensibilidad, menor precisión).")


# ------------------ Tab 3: Info ------------------
with tab_info:
    st.markdown("""
**Predicción de gravedad y mapa de riesgo de siniestros viales en Bogotá**

- **Datos:** 209.861 siniestros (2015–2021), Datos Abiertos Bogotá, licencia CC BY 4.0.
- **Modelo:** CatBoost multiclase (gravedad en 3 niveles), elegido por su recall en la clase fatal
  (~0,61 en el año de prueba 2019) frente a ~0,20 de otros modelos.
- **Determinantes (SHAP):** vulnerabilidad del usuario (ciclista, peatón, pasajero), causa y tipo de
  siniestro, vehículos pesados y momento del día.
- **Mapa:** tasa de fatalidad observada por zona (malla H3), robusta e independiente del modelo.

Repositorio: github.com/andresalmanzal/tfm-siniestros-bogota

*Autor: Ovidio Almanza Ledesma — Máster en Data Science, Big Data & Business Analytics (UCM).*

> El simulador es una herramienta orientativa de apoyo a la priorización, no un dictamen sobre casos reales.
""")
