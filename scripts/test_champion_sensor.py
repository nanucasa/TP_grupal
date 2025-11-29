import mlflow

EXPERIMENT_NAME = "telco_churn_tune_xgb"


def main():
    # Usa el mismo experimento que el sensor
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Creamos un run “falso” solo para probar el sensor,
    # con un f1 más alto que 0.6028
    with mlflow.start_run(run_name="sensor_test_run"):
        mlflow.log_metric("f1", 0.70)


if __name__ == "__main__":
    main()
