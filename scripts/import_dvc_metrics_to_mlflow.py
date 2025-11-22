import os
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

# Raíz del repo: C:\dvc_prueba
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Nombre del experimento donde SIEMPRE vamos a registrar todo
EXPERIMENT_NAME = "telco_churn_tune_xgb"

# Tracking URI de MLflow
# - Si tenés MLFLOW_TRACKING_URI en el entorno, se usa ese.
# - Si no, usa el remoto de DagsHub de este proyecto.
DEFAULT_DAGSHUB_MLFLOW_URI = "https://dagshub.com/nanucasa/TP_grupal.mlflow"


def get_tracking_uri() -> str:
    uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if uri_env:
        print(f"Usando MLFLOW_TRACKING_URI desde entorno: {uri_env}")
        return uri_env

    print(f"Usando tracking URI por defecto de DagsHub: {DEFAULT_DAGSHUB_MLFLOW_URI}")
    return DEFAULT_DAGSHUB_MLFLOW_URI


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def load_metrics_json(path: Path) -> dict:
    """
    Lee un JSON de métricas y devuelve solo pares clave:valor numéricos.
    Si hay diccionarios anidados, los aplana con nombre compuesto.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"El archivo {path} no contiene un dict de métricas.")

    metrics = {}

    def add_numeric(prefix: str, obj):
        if isinstance(obj, (int, float)):
            metrics[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}_{k}" if prefix else k
                add_numeric(new_key, v)
        # Si es lista u otro tipo, lo ignoramos para mantenerlo simple

    add_numeric("", data)

    # Limpiar posibles claves vacías
    metrics = {k: v for k, v in metrics.items() if k}

    return metrics


def main():
    # Configurar tracking remoto
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    # Asegurarnos de usar SIEMPRE este experimento
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        raise RuntimeError(
            f"No se pudo crear/obtener el experimento '{EXPERIMENT_NAME}' en {tracking_uri}"
        )

    print(f"Usando experimento '{EXPERIMENT_NAME}' (id={exp.experiment_id})")

    # Buscar todos los archivos metrics*.json en la raíz del repo
    metrics_files = sorted(PROJECT_ROOT.glob("metrics*.json"))

    if not metrics_files:
        print("No se encontraron archivos metrics*.json en la raíz del repo.")
        return

    print("Archivos de métricas encontrados:")
    for p in metrics_files:
        print(f"  - {p.name}")

    # Importar cada archivo como un run separado
    for path in metrics_files:
        try:
            metrics = load_metrics_json(path)
        except Exception as e:
            print(f"[ERROR] No se pudo leer {path.name}: {e}")
            continue

        if not metrics:
            print(f"[SKIP] {path.name} no tiene métricas numéricas para registrar.")
            continue

        run_name = path.stem  # p.ej. 'metrics_xgb', 'metrics_test_xgb', etc.

        print(f"[RUN] Creando run para {path.name} en '{EXPERIMENT_NAME}'...")

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=exp.experiment_id,
        ):
            # Métricas
            mlflow.log_metrics(metrics)

            # Tags útiles para filtrar en MLflow
            mlflow.set_tag("source", "dvc_metrics_json")
            mlflow.set_tag("metrics_file", str(path.name))

        print(f"[OK] Importado {path.name} en experimento '{EXPERIMENT_NAME}'")

    print("Importación de métricas finalizada.")


if __name__ == "__main__":
    main()
