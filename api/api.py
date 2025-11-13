from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import text
from .config import DB
from .models import init_db
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import os
from datetime import datetime
import traceback
import json
import io

app = FastAPI(title="API de Reentrenamiento - Regresión Logística")

# Inicializar tablas en Railway
init_db()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "columns.pkl")

# ----------------------------
# Modelo Pydantic
# ----------------------------
class DatosEntrada(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    loan: str
    y: int

# ----------------------------
# Guardar modelo en DB
# ----------------------------
def save_model(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    with DB.begin() as conn:
        conn.execute(
            text("INSERT INTO modelos (timestamp, modelo) VALUES (:ts, :modelo)"),
            {"ts": datetime.now(), "modelo": buffer.read()}
        )

# ----------------------------
# Cargar último modelo de DB
# ----------------------------
def load_latest_model():
    with DB.connect() as conn:
        row = conn.execute(
            text("SELECT modelo FROM modelos ORDER BY timestamp DESC LIMIT 1")
        ).fetchone()
        if row and row[0]:
            return joblib.load(io.BytesIO(row[0]))
        return None

# ----------------------------
# Función para reentrenar modelo
# ----------------------------
def retrain_model():
    try:
        df = pd.read_sql(text("SELECT * FROM insertar_datos"), DB.connect())
        if df.empty:
            print("⚠️ No hay datos para reentrenar.")
            return

        X = df.drop(columns=["y"])
        y = df["y"]

        if len(y.unique()) < 2:
            print("⚠️ No se puede reentrenar: solo hay una clase.")
            return

        X_encoded = pd.get_dummies(X, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)

        if os.path.exists(COLUMNS_PATH):
            saved_columns = joblib.load(COLUMNS_PATH)
            for col in saved_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[saved_columns]
        else:
            joblib.dump(X_encoded.columns, COLUMNS_PATH)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_encoded, y)

        # Guardar modelo en DB
        save_model(model)

        y_pred = model.predict(X_encoded)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        matriz_confusion = confusion_matrix(y, y_pred).tolist()
        pr_precision, pr_recall, _ = precision_recall_curve(y, model.predict_proba(X_encoded)[:, 1])

        with DB.begin() as conn:
            conn.execute(
                text("""
                     INSERT INTO metricas (timestamp, modelo, accuracy, precision, recall, f1, matriz_confusion, pr_precision, pr_recall)
                     VALUES (:timestamp, :modelo, :acc, :prec, :rec, :f1, :matriz_confusion, :pr_precision, :pr_recall)
                     """),
                {
                    "timestamp": datetime.now(),
                    "modelo": "Regresión Logística",
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "matriz_confusion": json.dumps(matriz_confusion),
                    "pr_precision": json.dumps(pr_precision.tolist()),
                    "pr_recall": json.dumps(pr_recall.tolist())
                }
            )
        print("✅ Métricas guardadas correctamente en metricas")

    except Exception as e:
        print("⚠️ Error al reentrenar:", e)
        print(traceback.format_exc())

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/insertar_datos/")
def insertar_datos(data: DatosEntrada, background_tasks: BackgroundTasks):
    try:
        with DB.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO insertar_datos (age, job, marital, education, balance, housing, loan, y)
                    VALUES (:age, :job, :marital, :education, :balance, :housing, :loan, :y)
                """),
                data.model_dump()
            )
        background_tasks.add_task(retrain_model)
        return {"message": "✅ Datos insertados y reentrenamiento iniciado."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/metricas/")
def get_metrics():
    df = pd.read_sql(text("SELECT * FROM metricas"), DB.connect())
    if not df.empty:
        df['timestamp'] = df['timestamp'].astype(str)
        for col in ['matriz_confusion', 'pr_precision', 'pr_recall']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and x not in [None, ""] else None)
        return df.to_dict(orient="records")
    return []

@app.post("/predecir/")
def predecir(data: DatosEntrada):
    try:
        modelo = load_latest_model()
        if modelo is None:
            return {"error": "No hay modelo entrenado aún"}

        columnas = joblib.load(COLUMNS_PATH)
        entrada = pd.DataFrame([data.model_dump()])
        entrada_encoded = pd.get_dummies(entrada, columns=["job", "marital", "education", "housing", "loan"], drop_first=True)

        for col in columnas:
            if col not in entrada_encoded.columns:
                entrada_encoded[col] = 0
        entrada_encoded = entrada_encoded[columnas]

        pred = modelo.predict(entrada_encoded)[0]
        prob = modelo.predict_proba(entrada_encoded)[0].tolist()

        return {"prediccion": int(pred), "probabilidades": prob}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/")
def home():
    return {
        "message": "✅ API de Reentrenamiento corriendo correctamente!",
        "endpoints": {
            "POST /insertar_datos/": "Inserta datos y reentrena el modelo",
            "GET /metricas/": "Obtiene métricas del modelo",
            "POST /predecir/": "Predice con el último modelo entrenado"
        }
    }
