# app.py
# ============================================================
# SNC-MACC (Versión Excel Upload)
# ============================================================

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# -----------------------------
# Config UI
# -----------------------------
st.set_page_config(
    page_title="SNC-MACC | Excel Upload",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 SNC-MACC — Carga desde Excel")
with open("plantilla_ejemplo.xlsx", "rb") as file:
    btn = st.download_button(
        label="📥 Descargar Plantilla Excel Vacía",
        data=file,
        file_name="plantilla_snc.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
st.caption("Sube tu plantilla Excel para generar la curva de abatimiento y portafolio.")
# -----------------------------
# 1. Lógica de lectura de EXCEL (NUEVO)
# -----------------------------
def safe_float(x, default=0.0) -> float:
    try:
        # Manejo de NaN de pandas o None
        if pd.isna(x) or x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def parse_excel_catalog(uploaded_file) -> List[dict]:
    """
    Lee un Excel con hojas 'Proyectos' y 'Flujos' y lo convierte
    a la estructura de lista de diccionarios que usa el motor de cálculo.
    """
    try:
        # Leemos las dos hojas
        df_proyectos = pd.read_excel(uploaded_file, sheet_name="Proyectos")
        df_flujos = pd.read_excel(uploaded_file, sheet_name="Flujos")
    except Exception as e:
        st.error(f"Error leyendo el Excel: {e}")
        st.stop()

    # Normalizamos nombres de columnas a minusculas para evitar errores de tipeo
    df_proyectos.columns = [c.strip().lower() for c in df_proyectos.columns]
    df_flujos.columns = [c.strip().lower() for c in df_flujos.columns]

    sncs_output = []

    # Iteramos por cada proyecto en la hoja principal
    for _, row in df_proyectos.iterrows():
        pid = str(row.get("id", "snc"))
        
        # Filtramos los flujos que pertenecen a este ID
        flows_df = df_flujos[df_flujos["id_proyecto"].astype(str) == pid]
        
        lines_list = []
        for _, f_row in flows_df.iterrows():
            l_type = str(f_row.get("tipo", "annual")).strip()
            val = safe_float(f_row.get("valor", 0.0))
            
            line_obj = {
                "nombre": f_row.get("nombre_linea", "Linea"),
                "categoria": f_row.get("categoria", "OTHER"),
                "tipo": l_type
            }

            # Configurar valores según tipo
            if l_type == "one_time":
                # Necesitamos saber en qué año
                year_app = int(safe_float(f_row.get("anio_aplicacion", 2027)))
                line_obj["valores_por_anio"] = {year_app: val}
            elif l_type == "annual":
                line_obj["valor_anual"] = val
            # carbon_linked no necesita valor, se calcula solo
            
            lines_list.append(line_obj)

        # Construimos el diccionario del proyecto
        snc_obj = {
            "id": pid,
            "nombre": row.get("nombre", pid),
            "area_ha_default": safe_float(row.get("area_ha", 1000)),
            "salvaguardas_pct_default": safe_float(row.get("salvaguardas_pct", 0)),
            "carbono": {
                "modelo": row.get("modelo_carbono", "sigmoid"),
                "tco2e_ha_total_default": safe_float(row.get("tco2e_ha_total", 100))
            },
            "finanzas": {
                "tasa_descuento_pct_default": safe_float(row.get("tasa_descuento_pct", 10)),
                "lineas": lines_list
            }
        }
        sncs_output.append(snc_obj)
        
    return sncs_output

# -----------------------------
# 2. Utilidades Generales (Casi igual que antes)
# -----------------------------
def years_range(start_year: int, end_year: int) -> List[int]:
    if end_year < start_year:
        return [start_year]
    return list(range(start_year, end_year + 1))

# -----------------------------
# Precio del carbono
# -----------------------------
def carbon_price_series(years: List[int], seed_year: int, seed_price: float, scenario: str, custom_series: Optional[Dict[int, float]] = None) -> pd.Series:
    years_sorted = sorted(years)
    if scenario == "Custom":
        if not custom_series: return pd.Series({y: seed_price for y in years_sorted})
        out = {}
        last = seed_price
        for y in years_sorted:
            if y in custom_series: last = float(custom_series[y])
            out[y] = last
        return pd.Series(out)
    
    growth_map = {"Constante": 0.00, "Bajo": 0.03, "Medio": 0.07, "Alto": 0.12}
    g = growth_map.get(scenario, 0.00)
    out = {}
    for y in years_sorted:
        if y < seed_year: out[y] = seed_price
        else:
            t = y - seed_year
            out[y] = seed_price * ((1 + g) ** t)
    return pd.Series(out)

# -----------------------------
# Modelos de carbono
# -----------------------------
def carbon_annual_curve_per_ha(years: List[int], model_type: str, tco2e_ha_total: float, sigmoid_r: float = 0.18, sigmoid_t_mid: float = 8.0, custom_series_per_ha: Optional[Dict[int, float]] = None) -> pd.Series:
    years_sorted = sorted(years)
    n = len(years_sorted)
    if n == 1: return pd.Series({years_sorted[0]: float(tco2e_ha_total)})
    
    model_type = (str(model_type) or "lineal").strip().lower() # cast str por seguridad

    if model_type in ["lineal", "linear"]:
        annual = float(tco2e_ha_total) / n
        return pd.Series({y: annual for y in years_sorted})

    # Sigmoide por defecto
    r = float(sigmoid_r)
    t_mid = float(sigmoid_t_mid)
    t = np.arange(1, n + 1, dtype=float)
    cum = 1 / (1 + np.exp(-r * (t - t_mid)))
    cum_norm = (cum - cum.min()) / (cum.max() - cum.min() + 1e-12)
    annual_frac = np.diff(np.r_[0.0, cum_norm])
    annual = annual_frac * float(tco2e_ha_total)
    return pd.Series({y: float(v) for y, v in zip(years_sorted, annual)})

def apply_carbon_adjustments(annual_tco2e: pd.Series, safeguard_pct: float) -> pd.Series:
    m = 1.0 * (1.0 - float(safeguard_pct) / 100.0)
    return annual_tco2e * m

# -----------------------------
# Finanzas
# -----------------------------
def build_cashflow_lines(years: List[int], lines: List[dict], carbon_revenue_series: Optional[pd.Series]) -> pd.DataFrame:
    records = []
    years_sorted = sorted(years)
    for ln in (lines or []):
        name = ln.get("nombre", "Línea")
        category = ln.get("categoria", "OTHER")
        ltype = (ln.get("tipo", "custom_series") or "custom_series").strip().lower()
        
        vals = {y: 0.0 for y in years_sorted}

        if ltype == "carbon_linked":
            if carbon_revenue_series is not None:
                vals = {int(y): float(carbon_revenue_series.get(y, 0.0)) for y in years_sorted}
        elif ltype == "annual":
            v = safe_float(ln.get("valor_anual", 0.0))
            vals = {y: v for y in years_sorted}
        elif ltype == "one_time":
            vby = ln.get("valores_por_anio", {}) or {}
            vals = {y: safe_float(vby.get(y, 0.0)) for y in years_sorted}
        
        for y in years_sorted:
            records.append({"year": int(y), "line_name": str(name), "category": str(category), "value_usd": float(vals[y])})
            
    df = pd.DataFrame(records)
    if df.empty: df = pd.DataFrame({"year": [], "line_name": [], "category": [], "value_usd": []})
    return df

def npv_from_series(net_cf: pd.Series, discount_rate: float, start_year: int) -> float:
    r = float(discount_rate)
    npv = 0.0
    for y, cf in net_cf.items():
        t = int(y) - int(start_year)
        npv += float(cf) / ((1 + r) ** t)
    return float(npv)

# -----------------------------
# Proyecto SNC: cálculo completo
# -----------------------------
def compute_project(snc: dict, global_start_year: int, global_end_year: int, area_override: Optional[float], price_series: pd.Series) -> dict:
    pid = snc.get("id", "snc")
    name = snc.get("nombre", pid)
    start_year = int(global_start_year)
    end_year = int(global_end_year)
    years = years_range(start_year, end_year)

    area_default = safe_float(snc.get("area_ha_default", 1000.0), 1000.0)
    area_ha = float(area_override) if area_override is not None else float(area_default)

    carb = snc.get("carbono", {}) or {}
    model = carb.get("modelo", "sigmoid")
    tco2e_ha_total = safe_float(carb.get("tco2e_ha_total_default", 100.0), 100.0)

    annual_per_ha = carbon_annual_curve_per_ha(years=years, model_type=model, tco2e_ha_total=tco2e_ha_total)
    annual_total = annual_per_ha * area_ha
    safeguard_pct = safe_float(snc.get("salvaguardas_pct_default", 0.0), 0.0)
    annual_net = apply_carbon_adjustments(annual_tco2e=annual_total, safeguard_pct=safeguard_pct)
    total_net = float(annual_net.sum())

    revenue_carbon = annual_net * price_series.reindex(annual_net.index).fillna(method="ffill").fillna(0.0)

    fin = snc.get("finanzas", {}) or {}
    discount_pct = safe_float(fin.get("tasa_descuento_pct_default", 10.0), 10.0)
    discount_rate = float(discount_pct) / 100.0
    lines = fin.get("lineas", []) or []

    df_lines = build_cashflow_lines(years=years, lines=lines, carbon_revenue_series=revenue_carbon)

    # Neto
    df_net = df_lines.groupby("year", as_index=False)["value_usd"].sum().sort_values("year")
    net_cf_series = pd.Series(df_net["value_usd"].values, index=df_net["year"].values)
    npv_with_carbon = npv_from_series(net_cf_series, discount_rate, start_year=start_year)

    # Neto sin ingreso carbono
    df_lines_no_carb = df_lines[~df_lines["line_name"].str.contains("carbon", case=False, regex=True)].copy()
    df_net_no_carb = df_lines_no_carb.groupby("year", as_index=False)["value_usd"].sum().sort_values("year")
    
    if df_net_no_carb.empty:
        npv_without_carbon = 0.0
    else:
        net_cf_series_nc = pd.Series(df_net_no_carb["value_usd"].values, index=df_net_no_carb["year"].values)
        npv_without_carbon = npv_from_series(net_cf_series_nc, discount_rate, start_year=start_year)

    mac_usd_per_t = np.inf
    if total_net > 0:
        mac_usd_per_t = float(npv_without_carbon) / float(total_net)

    return {
        "id": pid, "name": name, "area_ha": area_ha, "carbon_total_net": total_net,
        "mac_usd_per_t": mac_usd_per_t, "npv_with_carbon": npv_with_carbon,
        "npv_without_carbon": npv_without_carbon, "cashflow_lines": df_lines,
        "carbon_annual_net": annual_net
    }

# -----------------------------
# MACC y Graficación
# -----------------------------
def build_macc_dataframe(projects: List[dict]) -> pd.DataFrame:
    rows = []
    for p in projects:
        rows.append({
            "name": p["name"],
            "carbon_total_net_t": p["carbon_total_net"],
            "mac_usd_per_t": p["mac_usd_per_t"],
            "npv_without_carbon": p["npv_without_carbon"]
        })
    df = pd.DataFrame(rows)
    # Ordenar por MAC
    df["mac_sort"] = df["mac_usd_per_t"].replace([np.inf, -np.inf], np.nan)
    return df.sort_values("mac_sort").fillna(np.inf).reset_index(drop=True)

def plot_macc(df_macc: pd.DataFrame) -> go.Figure:
    if df_macc.empty: return go.Figure()
    
    df_plot = df_macc.copy()
    # Color logic
    df_plot["color"] = df_plot["mac_usd_per_t"].apply(lambda x: "#2ecc71" if x < 0 else ("#f39c12" if x <= 20 else "#e74c3c"))
    
    widths = df_plot["carbon_total_net_t"].values
    macs = df_plot["mac_usd_per_t"].values
    names = df_plot["name"].values
    colors = df_plot["color"].values

    cum_left = np.cumsum(np.r_[0.0, widths[:-1]])
    x_mid = cum_left + widths / 2.0
    
    fig = go.Figure(go.Bar(
        x=x_mid, y=macs, width=widths, text=names, marker_color=colors,
        hovertemplate="<b>%{text}</b><br>MAC: %{y:.2f}<br>Vol: %{width:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title="Curva de Abatimiento (MACC)",
        xaxis_title="Abatimiento acumulado (tCO₂e)",
        yaxis_title="Costo Marginal (USD/tCO₂e)",
        bargap=0, height=600
    )
    return fig

# -----------------------------
# Interfaz - Sidebar
# -----------------------------
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube archivo Excel (.xlsx)", type=["xlsx"])

st.sidebar.divider()
st.sidebar.header("2. Parámetros Globales")
start_year = st.sidebar.number_input("Año Inicio", 2027, 2050, 2027)
end_year = st.sidebar.number_input("Año Fin", 2028, 2060, 2040)
seed_price = st.sidebar.number_input("Precio Carbono Base (USD)", 0.0, 200.0, 30.0)
scenario = st.sidebar.selectbox("Escenario Precio", ["Constante", "Bajo", "Medio", "Alto"])

# -----------------------------
# Ejecución Principal
# -----------------------------

if uploaded_file is None:
    st.info("👋 **¡Bienvenidos!**")
    st.markdown("""
    Para empezar, por favor sube un archivo Excel con dos hojas:
    1. **Proyectos**: Definición de cada SNC (nombre, area, modelo carbono).
    2. **Flujos**: Costos e ingresos asociados (CAPEX, OPEX).
    
    Si no tienes el archivo, descarga la plantilla vacía abajo (simulada) o crea uno.
    """)
    st.stop()

# 1. Parsear Excel
sncs_list = parse_excel_catalog(uploaded_file)
st.success(f"✅ Se cargaron {len(sncs_list)} proyectos desde el Excel.")

# 2. Generar serie precios
years = years_range(start_year, end_year)
prices = carbon_price_series(years, start_year, seed_price, scenario)

# 3. Calcular Proyectos
projects_computed = []
for snc in sncs_list:
    # MVP: No overrides manuales de área en esta versión para simplificar, se usa lo del Excel
    p = compute_project(snc, start_year, end_year, None, prices)
    projects_computed.append(p)

# 4. MACC
df_macc = build_macc_dataframe(projects_computed)

st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Gráfica MACC")
    fig = plot_macc(df_macc)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Datos Tabla")
    st.dataframe(df_macc[["name", "mac_usd_per_t", "carbon_total_net_t"]].style.format({"mac_usd_per_t": "{:.2f}", "carbon_total_net_t": "{:,.0f}"}), use_container_width=True)

# 5. Descarga de resultados
csv = df_macc.to_csv(index=False).encode('utf-8')
st.download_button("Descargar Resultados (CSV)", csv, "macc_results.csv", "text/csv")



