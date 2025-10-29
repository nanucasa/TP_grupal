# scripts/mlflow_restore_experiment.py
# Restaura un experimento borrado en MLflow (backend SQLite) poniendo lifecycle_stage='active'.
import sqlite3, sys

DB_PATH = r"C:/dvc_prueba/mlflow.db"
EXP_NAME = "telco_churn_tune"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT experiment_id, name, lifecycle_stage, deleted_time FROM experiments WHERE name = ?", (EXP_NAME,))
row = cur.fetchone()
if not row:
    print(f"[ERROR] No existe experimento con nombre {EXP_NAME}")
    sys.exit(1)

exp_id, name, stage, deleted_time = row
if stage == "active":
    print(f"[OK] Ya estaba activo: id={exp_id}")
else:
    cur.execute("UPDATE experiments SET lifecycle_stage='active', deleted_time=NULL WHERE experiment_id=?", (exp_id,))
    conn.commit()
    print(f"[OK] Restaurado experimento id={exp_id} nombre={name}")

conn.close()
