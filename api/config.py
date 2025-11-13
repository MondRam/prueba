# api/config.py
from sqlalchemy import create_engine
import os

DB_SERVER = os.environ.get("DB_SERVER")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

DB = create_engine(
    f"mssql+pymssql://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}",
    fast_executemany=True
)                   # Cambia por tu contraseña

# -----------------------------
# Construir URL de conexión correcta
# -----------------------------
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

DB = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
