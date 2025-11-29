# src/train.py
# Entrena Logistic Regression, guarda artefactos y registra en MLflow

import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


def load_xy(path, target="churn"):
    df = pd.read_csv(path)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out-model", required=True)   # models\model_*.joblib
    ap.add_argument("--metrics", required=True)     # reports\metrics_*.json
    ap.add_argument(
        "--best-params",
        required=False,
        default=None,                               # artifacts\best_params.json (opcional)
        help="Ruta opcional a JSON con best_params; si no se pasa, se usan solo params.yaml",
    )
    ap.add_argument("--params", required=True)      # params.yaml
    ap.add_argument(
        "--run-name",
        required=False,
        default=None,
        help="Nombre legible del run en MLflow (ej: nanu_run_001).",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Hiperparámetros base desde params.yaml
    # ------------------------------------------------------------------
    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)

    C = float(P["train"]["C"])
    max_iter = int(P["train"]["max_iter"])
    seed = int(P["train"]["seed"])

    # ------------------------------------------------------------------
    # Override con best_params.json SOLO si se pasa y existe
    # ------------------------------------------------------------------
    if args.best_params is not None and os.path.exists(args.best_params):
        try:
            with open(args.best_params, "r", encoding="utf-8") as f:
                B = json.load(f)

            if "C" in B:
                C = float(B["C"])
            if "max_iter" in B:
                max_iter = int(B["max_iter"])
        except Exception:
            # Si algo falla leyendo best_params, seguimos con params.yaml
            pass

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------
    Xtr_df, ytr, _ = load_xy(args.train)
    Xva_df, yva, _ = load_xy(args.valid)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
            random_state=seed,
        )),
    ])

    # ------------------------------------------------------------------
    # Configuración de MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    )
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "telco_churn_tune_xgb")
    mlflow.set_experiment(exp_name)

    autor = "Nadia"

    # Forzar usuario también vía variables de entorno
    for var in ("LOGNAME", "USER", "USERNAME"):
        os.environ[var] = autor

    input_example = Xtr_df.head(5)
    run_name = args.run_name or "manual_run"

    # ------------------------------------------------------------------
    # Run de MLflow
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name=run_name) as run:
        # Forzar usuario/autor en tags y params
        mlflow.set_tag("mlflow.user", autor)   # User
        mlflow.set_tag("autor", autor)         # tag autor
        mlflow.log_param("autor", autor)       # param autor
        mlflow.log_param("run_name", run_name)

        pipe.fit(Xtr_df.to_numpy(dtype=float), ytr)
        prob = pipe.predict_proba(Xva_df.to_numpy(dtype=float))[:, 1]
        yhat = (prob >= 0.5).astype(int)

        valid_metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }

        # ------------------------------------------------------------------
        # Guardar artefactos locales
        # ------------------------------------------------------------------
        # out-model siempre tiene carpeta (ej: models\model.joblib)
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(pipe, args.out_model)

        # metrics puede NO tener carpeta (ej: "metrics.json" en el cwd)
        metrics_dir = os.path.dirname(args.metrics)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)

        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "valid": valid_metrics,
                    "params": {"C": C, "max_iter": max_iter, "seed": seed},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ------------------------------------------------------------------
        # Log de parámetros y métricas en MLflow
        # ------------------------------------------------------------------
        mlflow.log_params({"C": C, "max_iter": max_iter, "seed": seed})
        mlflow.log_metrics(
            {
                "accuracy": valid_metrics["accuracy"],
                "precision": valid_metrics["precision"],
                "recall": valid_metrics["recall"],
                "f1": valid_metrics["f1"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1": valid_metrics["f1"],
                "test_accuracy": valid_metrics["accuracy"],
                "test_precision": valid_metrics["precision"],
                "test_recall": valid_metrics["recall"],
                "test_f1": valid_metrics["f1"],
            }
        )

        signature = infer_signature(
            input_example,
            pipe.predict_proba(input_example.to_numpy(dtype=float))[:, 1],
        )

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        # Registrar en Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        reg_name = "TelcoChurn_LogReg"
        mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
        MlflowClient().transition_model_version_stage(
            name=reg_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        print(
            f"[OK {run_name}] valid: acc={valid_metrics['accuracy']:.4f} "
            f"prec={valid_metrics['precision']:.4f} "
            f"rec={valid_metrics['recall']:.4f} "
            f"f1={valid_metrics['f1']:.4f} "
            f"(C={C}, max_iter={max_iter}, seed={seed})"
        )


if __name__ == "__main__":
    main()

