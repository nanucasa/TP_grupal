import os
import json

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from dagster import asset, AssetExecutionContext

# ---------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------

EXPERIMENT_NAME = "telco_churn_tune_xgb"
MODEL_NAME = "TelcoChurn_XGB"
METRIC_NAME = "f1"

# Ruta al repo raíz: .../C:/dvc_prueba
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Carpeta y archivo donde guardamos la info del champion
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
CHAMPION_JSON_PATH = os.path.join(ARTIFACTS_DIR, "champion_from_mlflow.json")


# ---------------------------------------------------------------------
# 1) Buscar champion en MLflow
# ---------------------------------------------------------------------


@asset
def select_champion_from_mlflow(context: AssetExecutionContext) -> dict:
    """
    Busca en MLflow el mejor run del experimento definido en EXPERIMENT_NAME,
    ordenando por la métrica METRIC_NAME (f1).
    Devuelve un dict con toda la info relevante del champion.
    """
    tracking_uri = mlflow.get_tracking_uri()
    context.log.info(f"Usando MLflow tracking URI: {tracking_uri}")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        msg = f"No se encontró el experimento '{EXPERIMENT_NAME}' en MLflow."
        context.log.error(msg)
        raise RuntimeError(msg)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )

    if not runs:
        msg = (
            f"No se encontraron runs finalizados en el experimento "
            f"'{EXPERIMENT_NAME}'."
        )
        context.log.error(msg)
        raise RuntimeError(msg)

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(METRIC_NAME)

    context.log.info(
        f"Champion encontrado: run_id={best_run_id}, "
        f"{METRIC_NAME}={best_metric:.4f}"
    )

    result = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment.experiment_id,
        "run_id": best_run_id,
        "model_name": MODEL_NAME,
        "metric_name": METRIC_NAME,
        "metric_value": best_metric,
    }

    return result


# ---------------------------------------------------------------------
# 2) Persistir la info del champion en artifacts/champion_from_mlflow.json
# ---------------------------------------------------------------------


@asset
def persist_champion_json(
    context: AssetExecutionContext,
    select_champion_from_mlflow: dict,
) -> None:
    """
    Guarda el resultado de select_champion_from_mlflow en un JSON dentro de
    artifacts/champion_from_mlflow.json (en el repo raíz C:/dvc_prueba).
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(CHAMPION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(select_champion_from_mlflow, f, indent=2)

    context.log.info(
        f"Champion guardado en '{CHAMPION_JSON_PATH}': "
        f"{json.dumps(select_champion_from_mlflow, indent=2)}"
    )


# ---------------------------------------------------------------------
# 3) Intentar actualizar el alias 'champion' en la Model Registry
# ---------------------------------------------------------------------


@asset
def set_mlflow_champion_alias(
    context: AssetExecutionContext,
    select_champion_from_mlflow: dict,
) -> None:
    """
    Usa la info del champion para buscar la versión registrada del modelo
    MODEL_NAME con ese run_id y, si la encuentra, intenta ponerle el alias
    'champion'.

    Si el servidor de MLflow no permite actualizar aliases (por ejemplo,
    DagsHub devolviendo 403) se deja un warning en logs pero NO se rompe
    el pipeline.
    """
    tracking_uri = mlflow.get_tracking_uri()
    context.log.info(
        f"Intentando actualizar alias 'champion' en tracking URI: {tracking_uri}"
    )

    client = MlflowClient()
    run_id = select_champion_from_mlflow["run_id"]

    # 1) Buscar la versión de MODEL_NAME que corresponde a ese run_id
    try:
        model_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    except MlflowException as e:
        context.log.warning(
            f"No se pudieron leer las versiones registradas para el modelo "
            f"'{MODEL_NAME}'. Se omite la actualización del alias. "
            f"Detalle técnico: {e}"
        )
        return

    target_version = None
    for mv in model_versions:
        if getattr(mv, "run_id", None) == run_id:
            target_version = mv.version
            break

    if target_version is None:
        context.log.warning(
            f"No se encontró versión registrada de '{MODEL_NAME}' con "
            f"run_id={run_id}. Registra el modelo desde ese run en la UI de "
            f"MLflow y vuelve a ejecutar este asset si quieres tener alias."
        )
        return

    # 2) Intentar actualizar el alias 'champion'
    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", target_version)
        context.log.info(
            f"Alias 'champion' actualizado a {MODEL_NAME} v{target_version} "
            f"(run_id={run_id})."
        )
    except MlflowException as e:
        context.log.warning(
            "La llamada a la API de MLflow para actualizar el alias 'champion' "
            "falló (por ejemplo, error 403 en DagsHub). "
            "El pipeline continúa sin alias automático. "
            f"Detalle técnico: {e}"
        )
        return
