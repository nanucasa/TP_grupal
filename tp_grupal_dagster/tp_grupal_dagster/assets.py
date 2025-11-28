from pathlib import Path

import os
import json
import time

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dagster import MetadataValue, asset
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Ruta base del proyecto y carpeta de reportes
BASE_DIR = Path(r"C:\dvc_prueba")
REPORTS_DIR = BASE_DIR / "reports"

# MLflow remoto en DagsHub
REMOTE_TRACKING_URI = "https://dagshub.com/nanucasa/TP_grupal.mlflow"

# Experimentos de los que queremos leer las métricas de test
EXPERIMENT_NAMES = [
    "telco_churn_baseline",
    "telco_churn_baseline_fe",
    "telco_churn_baseline_rf",
]

# Métricas fijas de las curvas del modelo FE (las que se ven en los títulos de los plots)
PR_AP_FE = 0.598
ROC_AUC_FE = 0.742

# Experimento de tuning donde están los nanu_run_xxx
TUNE_EXPERIMENT_NAME = "telco_churn_tune_xgb"

# Métrica primaria para seleccionar champion
PRIMARY_METRIC = "test_f1"

# Nombre del modelo en el Model Registry
MODEL_NAME = "TelcoChurn_LogReg"

# Prefijo de los runs de Nadia
NANU_PREFIX = "nanu_run_"


@asset(
    description=(
        "Asset que junta las mejores métricas de test de los modelos entrenados "
        "leyendo directamente desde MLflow remoto en DagsHub."
    )
)
def test_metrics(context) -> pd.DataFrame:
    # Configuramos MLflow apuntando al tracking remoto de DagsHub
    mlflow.set_tracking_uri(REMOTE_TRACKING_URI)
    client = MlflowClient()

    rows = []

    for exp_name in EXPERIMENT_NAMES:
        try:
            exp = client.get_experiment_by_name(exp_name)
        except Exception as exc:
            context.log.warning(
                f"No se pudo leer el experimento '{exp_name}' desde MLflow remoto: {exc}"
            )
            continue

        if exp is None:
            context.log.warning(
                f"No se encontró el experimento '{exp_name}' en MLflow remoto."
            )
            continue

        # Buscamos los runs ordenados por f1_test descendente
        try:
            runs = client.search_runs(
                [exp.experiment_id],
                order_by=["metrics.test_f1 DESC"],
                max_results=1,
            )
        except Exception as exc:
            context.log.warning(
                f"Error al buscar runs del experimento '{exp_name}': {exc}"
            )
            continue

        if not runs:
            context.log.warning(
                f"El experimento '{exp_name}' no tiene runs con métricas."
            )
            continue

        best_run = runs[0]
        metrics = best_run.data.metrics
        params = best_run.data.params

        row = {
            "experiment_name": exp_name,
            "run_id": best_run.info.run_id,
            "model_name": params.get("model_name", exp_name),
            "f1_test": metrics.get("test_f1"),
            "precision_test": metrics.get("test_precision"),
            "recall_test": metrics.get("test_recall"),
            "roc_auc_test": metrics.get("test_roc_auc"),
            "pr_auc_test": metrics.get("test_pr_auc"),
        }
        rows.append(row)

    if not rows:
        context.log.warning(
            "No se pudo recuperar ninguna métrica desde MLflow remoto; "
            "se devuelve un DataFrame vacío."
        )
        df = pd.DataFrame(
            columns=[
                "experiment_name",
                "run_id",
                "model_name",
                "f1_test",
                "precision_test",
                "recall_test",
                "roc_auc_test",
                "pr_auc_test",
            ]
        )
    else:
        df = pd.DataFrame(rows)

    # Guardamos también un CSV en reports/ para tenerlo como artefacto
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "dagster_mlflow_test_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Metadata para Dagster (numérico + paths)
    context.add_output_metadata(
        {
            "num_modelos": MetadataValue.int(len(df)),
            "csv_path": MetadataValue.path(str(csv_path)),
            "tracking_uri": REMOTE_TRACKING_URI,
        }
    )

    return df


@asset(
    description=(
        "Gráfico de barras comparando el mejor F1 en test para cada modelo, "
        "según los datos de MLflow remoto."
    )
)
def f1_barchart(context, test_metrics: pd.DataFrame) -> str:
    # Validamos que haya datos para graficar
    if (
        test_metrics.empty
        or "model_name" not in test_metrics.columns
        or "f1_test" not in test_metrics.columns
    ):
        context.log.warning("No hay métricas F1 para graficar en f1_barchart.")
        return "Sin datos para F1."

    # Creamos el gráfico
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(test_metrics["model_name"], test_metrics["f1_test"])
    ax.set_xlabel("Modelo")
    ax.set_ylabel("F1 en test")
    ax.set_title("Comparación de F1 en test (MLflow remoto)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / "f1_comparison.png"
    fig.savefig(image_path)
    plt.close(fig)

    # Metadata: el path del archivo y un preview
    context.add_output_metadata(
        {
            "image_path": MetadataValue.path(str(image_path)),
        }
    )

    # Devolvemos la ruta de la imagen como valor del asset
    return str(image_path)


@asset(
    description=(
        "Registra en Dagster la curva PR (ya generada) del modelo FE. "
        "Lee la imagen desde C:/dvc_prueba/reports/pr_curve_fe.png."
    )
)
def pr_curve_fe(context, test_metrics: pd.DataFrame) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / "pr_curve_fe.png"

    if not image_path.exists():
        context.log.warning(
            f"No se encontró la imagen de PR curve FE en '{image_path}'."
        )

    # Intentamos, si existe, tomar pr_auc_test desde test_metrics para el modelo FE
    pr_auc_value = None
    if not test_metrics.empty and "pr_auc_test" in test_metrics.columns:
        try:
            mask = test_metrics["model_name"].str.contains("fe", case=False, na=False)
            fe_rows = test_metrics[mask]
            if not fe_rows.empty:
                pr_auc_value = float(fe_rows.iloc[0]["pr_auc_test"])
        except Exception:
            pr_auc_value = None

    # Si no encontramos nada en MLflow usamos el valor fijo que ya conocés
    if pr_auc_value is None:
        pr_auc_value = PR_AP_FE

    context.add_output_metadata(
        {
            "image_path": MetadataValue.path(str(image_path)),
            "pr_ap_fe": MetadataValue.float(pr_auc_value),
        }
    )

    return str(image_path)


@asset(
    description=(
        "Registra en Dagster la curva ROC (ya generada) del modelo FE. "
        "Lee la imagen desde C:/dvc_prueba/reports/roc_curve_fe.png."
    )
)
def roc_curve_fe(context, test_metrics: pd.DataFrame) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / "roc_curve_fe.png"

    if not image_path.exists():
        context.log.warning(
            f"No se encontró la imagen de ROC curve FE en '{image_path}'."
        )

    # Intentamos, si existe, tomar roc_auc_test desde test_metrics para el modelo FE
    roc_auc_value = None
    if not test_metrics.empty and "roc_auc_test" in test_metrics.columns:
        try:
            mask = test_metrics["model_name"].str.contains("fe", case=False, na=False)
            fe_rows = test_metrics[mask]
            if not fe_rows.empty:
                roc_auc_value = float(fe_rows.iloc[0]["roc_auc_test"])
        except Exception:
            roc_auc_value = None

    # Si no encontramos nada en MLflow usamos el valor fijo que ya conocés
    if roc_auc_value is None:
        roc_auc_value = ROC_AUC_FE

    context.add_output_metadata(
        {
            "image_path": MetadataValue.path(str(image_path)),
            "roc_auc_fe": MetadataValue.float(roc_auc_value),
        }
    )

    return str(image_path)


@asset(
    description=(
        "Selecciona el mejor run nanu_run_xxx del experimento telco_churn_tune_xgb "
        "según la métrica test_f1, actualiza artifacts/champion_run.json y el alias "
        "'champion' en el modelo TelcoChurn_LogReg."
    ),
)
def champion_run(context) -> dict:
    """Asset Dagster para elegir y registrar el champion."""
    # Apuntamos a MLflow remoto
    mlflow.set_tracking_uri(REMOTE_TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(TUNE_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(
            f"No se encontró el experimento {TUNE_EXPERIMENT_NAME!r} en MLflow remoto."
        )

    # Buscamos hasta 1000 runs activos
    runs = client.search_runs(
        [exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1000,
    )

    candidatos = []
    for r in runs:
        tags = r.data.tags
        run_name = tags.get("mlflow.runName", r.info.run_id)

        # Nos quedamos solo con los nanu_run_xxx
        if not run_name.startswith(NANU_PREFIX):
            continue

        metrics = r.data.metrics
        if PRIMARY_METRIC not in metrics:
            continue

        try:
            metric_value = float(metrics[PRIMARY_METRIC])
        except (TypeError, ValueError):
            continue

        candidatos.append((metric_value, r, run_name))

    if not candidatos:
        raise RuntimeError(
            f"No se encontraron runs con nombre {NANU_PREFIX!r} y métrica "
            f"{PRIMARY_METRIC!r} en el experimento {TUNE_EXPERIMENT_NAME!r}."
        )

    # Ordenamos por métrica descendente y tomamos el mejor
    candidatos.sort(key=lambda t: t[0], reverse=True)
    best_value, best_run, best_name = candidatos[0]

    champion_info = {
        "experiment_name": exp.name,
        "experiment_id": exp.experiment_id,
        "primary_metric": PRIMARY_METRIC,
        "run_id": best_run.info.run_id,
        "run_name": best_name,
        "metric_value": best_value,
    }

    # Guardamos artifacts/champion_run.json
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    champ_path = ARTIFACTS_DIR / "champion_run.json"
    with champ_path.open("w", encoding="utf-8") as f:
        json.dump(champion_info, f, ensure_ascii=False, indent=2)

    # Buscamos versión del modelo para este run
    version_for_run = None
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for mv in versions:
        if mv.run_id == best_run.info.run_id:
            version_for_run = int(mv.version)
            break

    aliases_payload = None
    if version_for_run is not None:
        # Aplicamos alias 'champion'
        client.set_registered_model_alias(MODEL_NAME, "champion", version_for_run)
        aliases_payload = {
            "applied": True,
            "alias": "champion",
            "name": MODEL_NAME,
            "version": version_for_run,
            "tracking_uri": REMOTE_TRACKING_URI,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        }
        alias_path = ARTIFACTS_DIR / "aliases_applied.json"
        with alias_path.open("w", encoding="utf-8") as f:
            json.dump(aliases_payload, f, ensure_ascii=False, indent=2)
    else:
        context.log.warning(
            f"No se encontró ninguna versión en {MODEL_NAME!r} con run_id "
            f"{best_run.info.run_id}; solo se actualizó champion_run.json.",
        )

    # Metadata para inspeccionar en Dagster
    meta = {
        "primary_metric": PRIMARY_METRIC,
        "metric_value": MetadataValue.float(best_value),
        "run_name": best_name,
        "run_id": best_run.info.run_id,
        "experiment_name": exp.name,
        "champion_json": MetadataValue.path(str(champ_path)),
    }
    if version_for_run is not None:
        meta["model_version"] = MetadataValue.int(version_for_run)

    context.add_output_metadata(meta)
    return champion_info


