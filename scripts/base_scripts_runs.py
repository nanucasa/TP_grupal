# scripts/base_scripts_runs.py

import os
from typing import Dict, Any, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


EXPERIMENT_NAME = "telco_churn_tune_xgb"
MODEL_NAME = "TelcoChurn_XGB"
METRIC_NAME = "f1"
AUTHOR = "Nadia"


def get_tracking_uris() -> Tuple[str, str]:
    """
    Devuelve (local_uri, remote_uri).

    - local_uri: siempre 'file:mlruns' en la raíz del repo.
    - remote_uri: se toma de la variable de entorno MLFLOW_TRACKING_URI (por ejemplo, DagsHub).
    """
    local_uri = "file:mlruns"
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")
    return local_uri, remote_uri


def get_or_create_experiment(tracking_uri: str) -> str:
    """
    Devuelve el experiment_id del experimento en un tracking_uri concreto.
    Si no existe, lo crea.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id
    return experiment_id


def load_data():
    """
    Carga los datos de churn desde data/processed/{train,valid}.csv.

    Usamos:
    - train.csv para entrenar
    - valid.csv para calcular el F1 que se loguea en MLflow
    """
    train_path = os.path.join("data", "processed", "train.csv")
    valid_path = os.path.join("data", "processed", "valid.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # IMPORTANTE: la columna target en tus datos se llama 'churn' en minúsculas
    target_col = "churn"

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])

    y_valid = valid_df[target_col]
    X_valid = valid_df.drop(columns=[target_col])

    return X_train, X_valid, y_train, y_valid


def train_and_evaluate(params: Dict[str, Any]) -> Tuple[XGBClassifier, float]:
    """
    Entrena el modelo con los hiperparámetros dados y devuelve (modelo, f1).
    """
    X_train, X_valid, y_train, y_valid = load_data()

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

    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    return model, f1


def log_single_run(
    tracking_uri: str,
    run_name: str,
    params: Dict[str, Any],
    model: XGBClassifier,
    f1: float,
) -> None:
    """
    Registra un run en UN servidor MLflow (tracking_uri).
    """
    if not tracking_uri:
        return

    experiment_id = get_or_create_experiment(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.set_tag("author", AUTHOR)
        mlflow.log_params(params)
        mlflow.log_metric(METRIC_NAME, f1)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )


def train_and_log_model_mirrored(
    run_name: str,
    params: Dict[str, Any],
) -> None:
    """
    Entrena UNA vez y refleja el run en:
      - MLflow local (file:mlruns)
      - MLflow remoto (MLFLOW_TRACKING_URI), si está definido.
    """
    local_uri, remote_uri = get_tracking_uris()

    # Entrenamos una sola vez
    model, f1 = train_and_evaluate(params)

    # Log en local
    log_single_run(
        tracking_uri=local_uri,
        run_name=run_name,
        params=params,
        model=model,
        f1=f1,
    )

    # Log en remoto (si existe y es distinto del local)
    if remote_uri and remote_uri != local_uri:
        log_single_run(
            tracking_uri=remote_uri,
            run_name=run_name,
            params=params,
            model=model,
            f1=f1,
        )


def main() -> None:
    # Algunos conjuntos de hiperparámetros de ejemplo
    param_grid: List[Dict[str, Any]] = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.10},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.08},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.12},
    ]

    for idx, params in enumerate(param_grid):
        run_name = f"nanu_run_{idx:03d}"
        train_and_log_model_mirrored(run_name=run_name, params=params)


if __name__ == "__main__":
    main()

