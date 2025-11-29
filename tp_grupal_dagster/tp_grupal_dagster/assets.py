# tp_grupal_dagster/assets.py

from __future__ import annotations

from dagster import asset, AssetIn
from pathlib import Path
from typing import Dict, Any
import json
import os

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


# ============================================================
# Constantes y paths compartidos
# ============================================================

# Raíz del proyecto Dagster: C:\dvc_prueba\tp_grupal_dagster
BASE_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivo donde guardamos el champion local
CHAMPION_METADATA_PATH = ARTIFACTS_DIR / "champion_metadata.json"

# Configuración del experimento / métrica de champion
EXPERIMENT_NAME = "telco_churn_tune_xgb"
MODEL_NAME = "TelcoChurn_XGB"
METRIC_NAME = "f1"


# ============================================================
# Helpers internos
# ============================================================

def _get_mlflow_client() -> MlflowClient:
    """
    Devuelve un MlflowClient usando MLFLOW_TRACKING_URI si está seteado.
    Si no, cae por defecto a .../mlruns relativo al repo.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        project_root = Path(__file__).resolve().parents[2]
        tracking_uri = f"file:///{project_root / 'mlruns'}"
    return MlflowClient(tracking_uri=tracking_uri)


def _get_best_run_from_mlflow(client: MlflowClient) -> Dict[str, Any]:
    """
    Busca en MLflow el mejor run del experimento definido en EXPERIMENT_NAME,
    ordenando por la métrica METRIC_NAME (f1).
    Devuelve un dict con toda la info relevante del champion.
    """
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experimento '{EXPERIMENT_NAME}' no encontrado en MLflow."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC_NAME} > 0",
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            f"No se encontraron runs válidos para el experimento '{EXPERIMENT_NAME}'."
        )

    run = runs[0]
    metric_value = run.data.metrics.get(METRIC_NAME)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "model_name": MODEL_NAME,
        "metric_name": METRIC_NAME,
        "metric_value": metric_value,
    }


# ============================================================
# Asset 1: elegir champion desde MLflow
# ============================================================

@asset
def select_champion_from_mlflow() -> Dict[str, Any]:
    """
    Consulta MLflow y devuelve la metadata del mejor run del experimento.
    """
    client = _get_mlflow_client()
    best = _get_best_run_from_mlflow(client)
    return best


# ============================================================
# Asset 2: persistir champion en JSON local
# ============================================================

@asset(ins={"champion": AssetIn(key="select_champion_from_mlflow")})
def persist_champion_json(champion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarda el champion en artifacts/champion_metadata.json.
    Este JSON es el que usa el sensor para saber cuál es el último champion persistido.
    """
    CHAMPION_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHAMPION_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(champion, f, indent=2, ensure_ascii=False)

    return champion


# ============================================================
# Asset 3: setear alias 'champion' en el Model Registry de MLflow
# ============================================================

@asset(ins={"champion": AssetIn(key="persist_champion_json")})
def set_mlflow_champion_alias(champion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asigna el alias 'champion' en el Model Registry de MLflow
    a la versión asociada al run campeón. Si no encuentra una versión
    registrada para ese run, la ejecución no falla.
    """
    client = _get_mlflow_client()

    model_name = champion["model_name"]
    run_id = champion["run_id"]

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        # Si no hay registry o falla la búsqueda, no rompemos el pipeline.
        return champion

    target_version: str | None = None
    for v in versions:
        if v.run_id == run_id:
            target_version = v.version
            break

    if target_version is None:
        # No encontramos una versión del modelo registrada con ese run_id.
        return champion

    # Intentamos setear alias 'champion'; si falla, no rompemos el pipeline.
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=target_version,
        )
    except MlflowException:
        pass

    return champion
