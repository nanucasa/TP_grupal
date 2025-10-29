# Borra PERMANENTEMENTE el experimento 'telco_churn_tune' de la BD SQLite de MLflow.
import sqlite3, os
DB = r"C:/dvc_prueba/mlflow.db"
NAME = "telco_churn_tune"

con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT experiment_id, name, lifecycle_stage FROM experiments WHERE name = ?", (NAME,))
row = cur.fetchone()
if not row:
    print("[OK] No existe en BD, nada que borrar."); con.close(); raise SystemExit(0)

eid, name, stage = row
print(f"[INFO] Encontrado: id={eid}, stage={stage}. Borrando fila...")
cur.execute("DELETE FROM experiments WHERE experiment_id=?", (eid,))
con.commit(); con.close()
print("[OK] Borrado permanente.")
