import numpy as np
import torch
import streamlit as st
from flame_in_freefall import FlamePredictor, insights, FEATURES, RISK_LABELS

st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")

st.title("🔥 Flame in Freefall")
st.caption("AI-Powered Fire Safety Insights from Microgravity Combustion Data")

@st.cache_resource
def load_predictor():
    return FlamePredictor("flame_net.pt")

predictor = load_predictor()

st.sidebar.header("Scenario Inputs")

cond = {
    "oxygen_conc": st.sidebar.slider("O2 concentration (%)", 12.0, 30.0, 21.0, 0.5),
    "pressure_kpa": st.sidebar.slider("Pressure (kPa)", 30.0, 150.0, 101.3, 1.0),
    "fuel_droplet_mm": st.sidebar.slider("Fuel droplet size (mm)", 0.5, 4.0, 1.0, 0.1),
    "ignition_energy_mj": st.sidebar.slider("Ignition energy (mJ)", 5.0, 200.0, 30.0, 5.0),
    "flow_velocity_cms": st.sidebar.slider("Airflow (cm/s)", 0.0, 10.0, 1.0, 0.5),
    "g_level": st.sidebar.slider("Residual gravity (ug)", 0.0, 50.0, 1.0, 0.5),
    "initial_temp_k": st.sidebar.slider("Ambient temperature (K)", 280.0, 340.0, 295.0, 1.0),
}

p = predictor.predict(cond)

RISK_COLORS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Fire Risk", f"{RISK_COLORS[p['fire_risk_class']]} {p['fire_risk_label']}")
    st.metric("Flame Spread", f"{p['flame_spread_rate_cms']:.3f} cm/s")
    st.metric("Soot Yield", f"{p['soot_yield']:.3f}")
    st.metric("Burn Duration", f"{p['burn_duration_s']:.1f} s")

with col2:
    st.subheader("Risk Probabilities")
    for i, label in enumerate(["LOW", "MODERATE", "HIGH", "CRITICAL"]):
        st.progress(float(p["fire_risk_probs"][i]),
                    text=f"{label}: {p['fire_risk_probs'][i]:.2%}")

st.subheader("Safety Insights")
for line in insights(p):
    st.write(f"• {line}")

st.caption("Reference: NASA CFM & FLEX microgravity combustion studies. "
           "Prototype only — not for operational use.")