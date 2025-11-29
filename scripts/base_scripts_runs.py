# scripts/run_nanu_runs.py

import os
from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


EXPERIMENT_NAME = "telco_churn_tune_xgb"
MODEL_NAME = "TelcoChurn_XGB"
METRIC_NAME = "f1"


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Devuelve el experiment_id del experimento. Si no existe, lo crea.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = exp.experiment_id
    return experiment_id


def load_data():
    """
    Carga el dataset de churn y lo separa en train / test.

    AJUSTÁ:
    - La ruta del CSV si en tu repo es distinta.
    - El nombre de la columna target si no se llama 'Churn'.
    """
    df = pd.read_csv("data/processed/telco_churn_prepared.csv")

    y = df["Churn"]          # <- cambia si tu target tiene otro nombre
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_and_log_model(
    run_name: str,
    params: Dict[str, Any],
    author: str = "Nadia",
) -> None:
    """
    Entrena un XGBClassifier con 'params' y lo registra en MLflow,
    incluido el Model Registry (MODEL_NAME).
    """
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=run_name):
        # Tag para filtrar por autora
        mlflow.set_tag("author", author)

        # Log de hiperparámetros
        mlflow.log_params(params)

        model = XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        model.fit(X_train, y_train)

        # Métrica principal (la que usa Dagster para el champion)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric(METRIC_NAME, f1)

        # REGISTRO DEL MODELO EN EL MODEL REGISTRY
        # (esto crea/actualiza versiones de 'TelcoChurn_XGB' asociadas al run_id)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )


def main():
    # Usa el tracking URI que tengas configurado (local o DagsHub)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_id=experiment_id)

    # Algunos conjuntos de hiperparámetros de ejemplo
    param_grid: List[Dict[str, Any]] = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.10},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.08},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.12},
    ]

    for idx, params in enumerate(param_grid):
        run_name = f"nanu_run_{idx:03d}"
        train_and_log_model(run_name=run_name, params=params, author="Nadia")


if __name__ == "__main__":
    main()
