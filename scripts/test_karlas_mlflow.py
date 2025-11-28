import os
import mlflow


def main():
    # Forzar que en la MLflow UI el usuario se vea como "Karla"
    for var in ("LOGNAME", "USER", "USERNAME"):
        os.environ[var] = "Karla"

    # Tracking server: tu DagsHub (nanucasa) o el que esté en el entorno
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/nanucasa/TP_grupal.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Experimento por defecto
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco_churn_tune_xgb")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="karlas_run_XXX"):
        # Marca clara de que este run es de Karla
        mlflow.log_param("Author", "Karla")   # en DagsHub se verá como columna params.Author
        mlflow.set_tag("author", "Karla")     # tag útil para filtrar en la MLflow UI

        # =====================================================
        # TODO: reemplazar por tu entrenamiento REAL
        # Ejemplo mínimo mientras tanto:
        mlflow.log_metric("metric_test_karla", 0.999)
        # =====================================================


if __name__ == "__main__":
    main()
