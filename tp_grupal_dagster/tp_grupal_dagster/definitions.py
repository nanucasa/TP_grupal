from dagster import (
    Definitions,
    load_assets_from_modules,
    fs_io_manager,
    AssetSelection,
    define_asset_job,
    sensor,
    RunRequest,
    SensorEvaluationContext,
)

import mlflow

from . import assets

# -------------------------------------------------------------------
# Configuración del experimento / métrica de champion
# -------------------------------------------------------------------
EXPERIMENT_NAME = "telco_churn_tune_xgb"
METRIC_NAME = "f1"

# -------------------------------------------------------------------
# Assets
# -------------------------------------------------------------------
all_assets = load_assets_from_modules([assets])

# Job que ejecuta SOLO el flujo de champion
champion_job = define_asset_job(
    name="champion_job",
    selection=AssetSelection.assets(
        assets.select_champion_from_mlflow,
        assets.persist_champion_json,
        assets.set_mlflow_champion_alias,
    ),
)

# -------------------------------------------------------------------
# Sensor: lanza champion_job SOLO cuando cambia el champion en MLflow
# -------------------------------------------------------------------
@sensor(job=champion_job)
def champion_sensor(context: SensorEvaluationContext):
    """
    Sensor que:
    - Busca el mejor run por METRIC_NAME en el experimento EXPERIMENT_NAME.
    - Compara su run_id con context.cursor.
    - Si cambió, dispara el job y actualiza el cursor.
    """

    client = mlflow.tracking.MlflowClient()

    # 1) Buscar experimento en MLflow
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        context.log.warning(
            f"Experimento '{EXPERIMENT_NAME}' no encontrado en MLflow. "
            "El sensor no disparará el job."
        )
        return

    # 2) Buscar mejor run por métrica
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )

    if not runs:
        context.log.info(
            f"No hay runs en el experimento '{EXPERIMENT_NAME}'. "
            "El sensor no disparará el job."
        )
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(METRIC_NAME)

    last_seen_run_id = context.cursor  # puede ser None la primera vez

    # 3) Si el champion no cambió, no hacemos nada
    if last_seen_run_id == best_run_id:
        context.log.info(
            f"Champion sin cambios "
            f"(run_id={best_run_id}, {METRIC_NAME}={best_metric:.4f}). "
            "No se dispara champion_job."
        )
        return

    # 4) Nuevo champion detectado -> disparamos el job y actualizamos cursor
    context.log.info(
        f"Nuevo champion detectado: run_id={best_run_id}, "
        f"{METRIC_NAME}={best_metric:.4f}. "
        f"Anterior run_id visto: {last_seen_run_id!r}. "
        "Se dispara champion_job."
    )

    # Actualizamos cursor ANTES de lanzar el run para evitar duplicados
    context.update_cursor(best_run_id)

    # 5) Disparar run del job
    yield RunRequest(
        run_key=best_run_id,  # evita correr dos veces para el mismo champion
        tags={
            "mlflow_experiment": EXPERIMENT_NAME,
            "mlflow_champion_run_id": best_run_id,
        },
    )


# -------------------------------------------------------------------
# Definitions de Dagster
# -------------------------------------------------------------------
defs = Definitions(
    assets=all_assets,
    jobs=[champion_job],
    sensors=[champion_sensor],
    resources={"io_manager": fs_io_manager},
)
