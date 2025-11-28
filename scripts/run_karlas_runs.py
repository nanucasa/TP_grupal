# scripts/run_karlas_runs.py
# Script genérico para lanzar un run en MLflow sobre el experimento telco_churn_tune_xgb

import os
import argparse
import mlflow


def main():
    parser = argparse.ArgumentParser(description="Lanza un run en MLflow para telco_churn_tune_xgb.")
    parser.add_argument(
        "--author",
        default="desconocido",
        help="Nombre de quien ejecuta el experimento (se guarda como tag 'author').",
    )
    parser.add_argument(
        "--run-name",
        default="manual_run",
        help="Nombre legible del run en MLflow.",
    )
    args = parser.parse_args()

    # Tracking server: se toma del entorno, con un default razonable
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/nanucasa/TP_grupal.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Experimento por defecto (el que ya venís usando)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco_churn_tune_xgb")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=args.run_name):
        # Tag genérico para saber quién corrió el experimento
        mlflow.set_tag("author", args.author)

        # TODO: acá iría la llamada al entrenamiento real
        # Ejemplo mínimo mientras tanto:
        mlflow.log_metric("metric_test", 0.999)


if __name__ == "__main__":
    main()
