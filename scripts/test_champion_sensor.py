import os
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Nombre del experimento y modelo que usamos en todo el TP
EXPERIMENT_NAME = "telco_churn_tune_xgb"
MODEL_NAME = "TelcoChurn_XGB"

# Ruta del JSON de champion: la MISMA que usa Dagster
# __file__ -> C:\dvc_prueba\scripts\test_champion_sensor.py
# parents[1] -> C:\dvc_prueba
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = PROJECT_ROOT / "tp_grupal_dagster" / "artifacts" / "champion_metadata.json"


def get_best_run(client: MlflowClient, experiment_id: str):
    """
    Devuelve un dict con la info del mejor run (por f1) o None si no hay.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1 DESC"],  # ordena de mayor a menor f1
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    f1 = run.data.metrics.get("f1")
    if f1 is None:
        return None

    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
        "model_name": MODEL_NAME,
        "metric_name": "f1",
        "metric_value": f1,
    }


def load_local_champion(path: Path):
    """
    Carga el JSON local del champion si existe, sino devuelve None.
    """
    if not path.exists():
        print(f"[INFO] No existe {path}, no hay champion local guardado.")
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_json(label: str, data):
    if data is None:
        print(f"[INFO] {label}: None")
        return
    print(f"[INFO] {label}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    # Mostrar y usar el tracking URI actual
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"[INFO] MLFLOW_TRACKING_URI = {tracking_uri}")
    if not tracking_uri:
        print("[ERROR] MLFLOW_TRACKING_URI no está definido.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Buscar el experimento por nombre
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"[ERROR] Experimento '{EXPERIMENT_NAME}' no encontrado.")
        return

    print(f"[INFO] Experiment id = {exp.experiment_id}")

    # Mejor run en MLflow
    best = get_best_run(client, exp.experiment_id)
    if best is None:
        print("[RESULT] No hay runs con métrica f1. El sensor no dispararía nada.")
        return

    pretty_print_json("Mejor run actual en MLflow", best)

    # Champion local (el que persiste Dagster)
    local = load_local_champion(CHAMPION_PATH)
    pretty_print_json("Champion local actual", local)

    # Lógica del "sensor"
    if local is None:
        print(
            "[RESULT] No hay champion local guardado. "
            "El sensor dispararía el job de actualización de champion."
        )
        return

    if local.get("run_id") == best["run_id"]:
        print(
            "[RESULT] El champion local YA corresponde al mejor run de MLflow. "
            "El sensor NO dispararía el job."
        )
        return

    if local.get("metric_value", 0) >= best["metric_value"]:
        print(
            "[RESULT] El champion local tiene un f1 >= al mejor run actual. "
            "El sensor NO dispararía el job."
        )
        return

    print(
        "[RESULT] Hay un run en MLflow con mejor f1 que el champion local. "
        "El sensor dispararía el job de actualización de champion."
    )


if __name__ == "__main__":
    main()
