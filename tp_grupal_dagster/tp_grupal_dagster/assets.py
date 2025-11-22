from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dagster import MetadataValue, asset
from mlflow.tracking import MlflowClient

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

        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="metrics.f1_test IS NOT NULL",
                order_by=["metrics.f1_test DESC"],
                max_results=1,
            )
        except Exception as exc:
            context.log.warning(
                f"Error buscando runs del experimento '{exp_name}': {exc}"
            )
            continue

        if not runs:
            context.log.warning(
                f"El experimento '{exp_name}' no tiene runs con métrica 'f1_test'."
            )
            continue

        best_run = runs[0]
        metrics = best_run.data.metrics
        tags = best_run.data.tags

        # Nombre de modelo: usamos tag si existe, sino el nombre del experimento
        model_name = tags.get("model_name", exp.name)

        def _get_metric(name: str):
            """Devuelve la métrica como float si existe, sino None."""
            if name not in metrics:
                return None
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                return None

        rows.append(
            {
                "experiment_name": exp.name,
                "run_id": best_run.info.run_id,
                "model_name": model_name,
                "f1_test": _get_metric("f1_test"),
                "precision_test": _get_metric("precision_test"),
                "recall_test": _get_metric("recall_test"),
                "roc_auc_test": _get_metric("roc_auc_test"),
                "pr_auc_test": _get_metric("pr_auc_test"),
            }
        )

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
    ax.set_ylabel("F1 score")
    ax.set_title("Comparación F1 en test")
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REPORTS_DIR / "f1_bench_dagster.png"
    fig.savefig(image_path)
    plt.close(fig)

    # Metadata: path de la imagen + un valor numérico por modelo
    meta = {
        "image_path": MetadataValue.path(str(image_path)),
    }

    for _, row in test_metrics.iterrows():
        model_name = str(row["model_name"])
        f1_value = row.get("f1_test")
        try:
            f1_float = float(f1_value)
        except (TypeError, ValueError):
            continue

        # Esto genera una serie por modelo en la pestaña "Plots"
        key = f"f1_test_{model_name}"
        meta[key] = MetadataValue.float(f1_float)

    context.add_output_metadata(meta)

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
            "average_precision_fe": MetadataValue.float(pr_auc_value),
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

