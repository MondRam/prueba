import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ttest_1samp
import json
import numpy as np
import os

API_URL = os.getenv("API_URL")

st.title("🤖 Dashboard - Modelo Logístico")

# -----------------------------
# 🔄 Insertar registro y predecir
# -----------------------------
st.header("🧾 Insertar registro y obtener predicción")

with st.form("formulario_unico"):
    age = st.number_input("Edad", 18, 100)
    job = st.selectbox("Ocupación", ["admin.","blue-collar","technician","services","management"])
    marital = st.selectbox("Estado civil", ["single","married","divorced"])
    education = st.selectbox("Educación", ["primary","secondary","tertiary"])
    balance = st.number_input("Balance", -5000, 100000)
    housing = st.selectbox("Hipoteca", ["yes","no"])
    loan = st.selectbox("Préstamo", ["yes","no"])
    y = st.selectbox("Aceptó producto (histórico)", [0,1])
    submitted = st.form_submit_button("Guardar y predecir")

if submitted:
    payload_insert = {
        "age": age, "job": job, "marital": marital, "education": education,
        "balance": balance, "housing": housing, "loan": loan, "y": y
    }

    payload_pred = {
        "age": age, "job": job, "marital": marital, "education": education,
        "balance": balance, "housing": housing, "loan": loan
    }

    res_insert = requests.post(f"{API_URL}/insertar_datos/", json=payload_insert)
    if res_insert.ok:
        st.success("✅ Registro guardado y reentrenamiento disparado.")
    else:
        st.error(f"❌ Error al insertar: {res_insert.text}")

    res_pred = requests.post(f"{API_URL}/predecir/", json=payload_pred)
    if res_pred.ok:
        resultado = res_pred.json()
        if "prediccion" in resultado:
            st.success(f"🔮 Predicción: {resultado['prediccion']}")
            st.write("Probabilidades:", resultado["probabilidades"])
        elif "error" in resultado:
            st.error(f"❌ Error en predicción: {resultado['error']}")
            st.text(resultado.get("trace", ""))
        else:
            st.warning("⚠️ Respuesta inesperada del servidor")
    else:
        st.error(f"❌ Error en predicción: {res_pred.text}")

# -----------------------------
# 📈 Métricas del modelo
# -----------------------------
st.header("📊 Métricas del modelo")

try:
    res = requests.get(f"{API_URL}/metricas/")
    if res.ok:
        data = res.json()
        if data:
            df = pd.DataFrame(data)
            st.subheader("Tabla Histórica")
            st.dataframe(df)

            chart_df = df[["timestamp","accuracy","precision","recall","f1"]].set_index("timestamp")
            st.line_chart(chart_df)

            cm = df["matriz_confusion"].iloc[-1]
            if cm is not None:
                cm