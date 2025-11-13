import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ttest_1samp
import json
import numpy as np
import os

API_URL = os.getenv("API_URL")  # Define esta variable en Railway con la URL pública de tu API

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
    y = st.selectbox("Aceptó producto", [0,1])
    submitted = st.form_submit_button("Guardar y predecir")

    if submitted:
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "balance": balance, "housing": housing, "loan": loan, "y": y
        }

        # Insertar en la base y reentrenar
        res_insert = requests.post(f"{API_URL}/insertar_datos/", json=payload)
        if res_insert.ok:
            st.success("✅ Registro guardado y modelo reentrenado.")
        else:
            st.error(f"❌ Error al insertar: {res_insert.text}")

        # Pedir predicción
        res_pred = requests.post(f"{API_URL}/predecir/", json=payload)
        if res_pred.ok:
            resultado = res_pred.json()
            st.success(f"🔮 Predicción: {resultado['prediccion']}")
            st.write("Probabilidades:", resultado["probabilidades"])
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

            # Tabla histórica
            st.subheader("Tabla Histórica")
            st.dataframe(df)

            # Gráfica de métricas
            chart_df = df[["timestamp","accuracy","precision","recall","f1"]].set_index("timestamp")
            st.line_chart(chart_df)

            # Última matriz de confusión
            cm = df["matriz_confusion"].iloc[-1]
            if cm is not None:
                cm = np.array(cm)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm).plot(ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ No hay matriz de confusión disponible")

            # Curva Precision-Recall
            pr_precision = df["pr_precision"].iloc[-1]
            pr_recall = df["pr_recall"].iloc[-1]
            if pr_precision and pr_recall:
                fig, ax = plt.subplots()
                ax.plot(pr_recall, pr_precision, marker='.')
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Curva Precision-Recall")
                st.pyplot(fig)
            else:
                st.warning("⚠️ No hay datos de Precision-Recall disponibles")

            # Prueba de hipótesis (accuracy > 0.9)
            accuracy_vals = df["accuracy"].astype(float)
            t_stat, p_val = ttest_1samp(accuracy_vals, 0.9)
            alpha = 0.05
            if p_val/2 < alpha and t_stat > 0:
                st.success("✅ Rechazamos H0: el modelo ha mejorado significativamente")
            else:
                st.warning("⚠️ No se puede rechazar H0")
        else:
            st.warning("⚠️ No hay métricas registradas aún")
    else:
        st.error(f"❌ Error al obtener métricas: {res.status_code}")
except Exception as e:
    st.error(f"❌ Error al procesar métricas: {e}")
