# api/config.py
from sqlalchemy import create_engine
import urllib

# -----------------------------
# CONFIGURACIÓN DE LA BASE DE DATOS
# -----------------------------
DB_SERVER = "mlpserver.database.windows.net"  # Cambia por tu servidor
DB_NAME = "ml_db"                             # Cambia por tu base de datos
DB_USER = "mladmin"                            # Cambia por tu usuario
DB_PASSWORD = "Equipo269"                      # Cambia por tu contraseña

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
