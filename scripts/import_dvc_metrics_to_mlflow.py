import json
from pathlib import Path
import mlflow


def filename_to_experiment(fname: str) -> str:
    """Mapea cada metrics_*.json a un experimento MLflow."""
    mapping = {
        "metrics.json": "telco_churn_baseline",
        "metrics_fe.json": "telco_churn_baseline_fe",
        "metrics_rf.json": "telco_churn_baseline_rf",
        "metrics_xgb.json": "telco_churn_baseline_xgb",
        "metrics_test.json": "telco_churn_test",
        "metrics_test_fe.json": "telco_churn_test_fe",
        "metrics_test_rf.json": "telco_churn_test_rf",
        "metrics_test_xgb.json": "telco_churn_test_xgb",
        "metrics_tune.json": "telco_churn_tune",
        "metrics_tune_rf.json": "telco_churn_tune_rf",
        "metrics_tune_xgb.json": "telco_churn_tune_xgb",
        "metrics_threshold.json": "telco_churn_threshold",
        "metrics_threshold_fe.json": "telco_churn_threshold_fe",
        "metrics_threshold_xgb.json": "telco_churn_threshold_xgb",
    }
    return mapping.get(fname, "telco_churn_imported")


def log_one_file(path: Path) -> None:
    fname = path.name
    experiment_name = filename_to_experiment(fname)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Usa el tracking URI definido en las variables de entorno
    mlflow.set_experiment(experiment_name)

    # Un run por archivo, identificado por el nombre del archivo
    with mlflow.start_run(run_name=fname):
        # métrica de validación
        if "valid" in data and isinstance(data["valid"], dict):
            mlflow.log_metrics(
                {f"valid_{k}": float(v) for k, v in data["valid"].items()}
            )

        # métrica de test
        if "test" in data and isinstance(data["test"], dict):
            mlflow.log_metrics(
                {f"test_{k}": float(v) for k, v in data["test"].items()}
            )

        # métricas de búsqueda de threshold (si existen)
        if "valid_threshold" in data and isinstance(data["valid_threshold"], dict):
            mlflow.log_metrics(
                {
                    f"valid_threshold_{k}": float(v)
                    for k, v in data["valid_threshold"].items()
                }
            )

        # best_valid_f1 (tune / tune_xgb)
        if "best_valid_f1" in data:
            mlflow.log_metric("best_valid_f1", float(data["best_valid_f1"]))

        # parámetros base
        if "params" in data and isinstance(data["params"], dict):
            mlflow.log_params({str(k): v for k, v in data["params"].items()})

        # best_params (tuning RF / XGB / LogReg)
        if "best_params" in data and isinstance(data["best_params"], dict):
            mlflow.log_params(
                {f"best_{k}": v for k, v in data["best_params"].items()}
            )

        # grid de búsqueda, lo guardo como JSON en un solo parámetro
        if "grid" in data and isinstance(data["grid"], dict):
            mlflow.log_param("grid", json.dumps(data["grid"]))

        print(f"[OK] Importado {fname} en experimento '{experiment_name}'")


def main():
    # Asumimos que corrés el script desde la raíz del repo (C:\\dvc_prueba)
    base = Path(".")
    for path in sorted(base.glob("metrics*.json")):
        log_one_file(path)


if __name__ == "__main__":
    main()
