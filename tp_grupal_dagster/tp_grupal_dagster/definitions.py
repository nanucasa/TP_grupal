# tp_grupal_dagster/definitions.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

from dagster import (
    AssetSelection,
    Definitions,
    RunRequest,
    SkipReason,
    SensorEvaluationContext,
    define_asset_job,
    load_assets_from_modules,
    sensor,
)

from mlflow.tracking import MlflowClient

from . import assets


# ============================================================
# Helpers compartidos con el sensor
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


def _load_local_champion(path: Path) -> Optional[Dict[str, Any]]:
    """
    Carga el champion local desde el JSON en artifacts.
    Devuelve None si no existe o si hay error al leerlo.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_new_champion(
    remote_best: Dict[str, Any],
    local: Optional[Dict[str, Any]],
) -> bool:
    """
    True si el run_id remoto es distinto al run_id del champion local,
    o si no hay champion local.
    """
    if local is None:
        return True
    return remote_best.get("run_id") != local.get("run_id")


# ============================================================
# Dagster assets + job
# ============================================================

# Cargamos todos los assets definidos en assets.py
all_assets = load_assets_from_modules([assets])

# Job que ejecuta SOLO el flujo de champion
champion_job = define_asset_job(
    name="champion_job",
    selection=AssetSelection.keys(
        "select_champion_from_mlflow",
        "persist_champion_json",
        "set_mlflow_champion_alias",
    ),
)


# ============================================================
# Sensor: lanza champion_job SOLO cuando cambia el champion en MLflow
# ============================================================

@sensor(job=champion_job)
def champion_sensor(context: SensorEvaluationContext):
    """
    Sensor que:
    - Consulta MLflow y obtiene el mejor run por METRIC_NAME del experimento EXPERIMENT_NAME.
    - Compara su run_id con el champion local guardado en artifacts/champion_metadata.json.
    - Si cambió, dispara el job de assets de champion.
    """

    # Mejor run actual en MLflow
    client = _get_mlflow_client()
    remote_best = assets._get_best_run_from_mlflow(client)  # reutilizamos lógica de assets

    context.log.info("Mejor run actual en MLflow:")
    context.log.info(json.dumps(remote_best, indent=2, ensure_ascii=False))

    # Champion local (JSON en artifacts)
    local_champion = _load_local_champion(assets.CHAMPION_METADATA_PATH)
    if local_champion is None:
        context.log.info(
            f"No existe {assets.CHAMPION_METADATA_PATH}, no hay champion local guardado."
        )
    else:
        context.log.info("Champion local actual:")
        context.log.info(json.dumps(local_champion, indent=2, ensure_ascii=False))

    # ¿Hay nuevo champion?
    if not _is_new_champion(remote_best, local_champion):
        context.log.info(
            "No se detectó un nuevo champion. El sensor no disparará ningún job."
        )
        yield SkipReason("No hay nuevo champion en MLflow.")
        return

    # Nuevo champion -> disparamos el job
    context.log.info(
        f"Nuevo champion detectado: run_id={remote_best['run_id']}, "
        f"{assets.METRIC_NAME}={remote_best['metric_value']}"
    )
    context.log.info("El sensor disparará materialización de assets de champion.")

    # Disparamos el job de assets de champion
    yield RunRequest(
        run_key=remote_best["run_id"],
        tags={
            "mlflow_experiment_id": remote_best["experiment_id"],
            "mlflow_run_id": remote_best["run_id"],
        },
    )


# ============================================================
# Definitions raíz para Dagster
# ============================================================

defs = Definitions(
    assets=all_assets,
    jobs=[champion_job],
    sensors=[champion_sensor],
)
