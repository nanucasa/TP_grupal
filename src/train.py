# src/train.py
# Entrena Logistic Regression con mejores hiperparámetros, loguea y registra en MLflow local con signature + input_example
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, mlflow, mlflow.sklearn, yaml
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
    ap.add_argument("--out-model", required=True)   # models\model.joblib
    ap.add_argument("--metrics", required=True)     # metrics.json
    ap.add_argument("--best-params", required=True) # artifacts\best_params.json
    ap.add_argument("--params", required=True)      # params.yaml
    args = ap.parse_args()

    # Hiperparámetros base (params.yaml) + override por best_params.json
    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)
    C = float(P["train"]["C"])
    max_iter = int(P["train"]["max_iter"])
    seed = int(P["train"]["seed"])

    if os.path.exists(args.best_params):
        try:
            B = json.load(open(args.best_params, "r", encoding="utf-8"))
            C = float(B.get("C", C))
            max_iter = int(B.get("max_iter", max_iter))
        except Exception:
            pass

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

    # MLflow local
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_baseline")

    # Input example + signature (usa columnas tal cual del train)
    input_example = Xtr_df.head(5)
    with mlflow.start_run() as run:
        pipe.fit(Xtr_df.to_numpy(dtype=float), ytr)
        prob = pipe.predict_proba(Xva_df.to_numpy(dtype=float))[:, 1]
        yhat = (prob >= 0.5).astype(int)

        # Métricas en validación
        valid_metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }

        # Guardar artefactos locales (mantenemos el esquema anterior de metrics.json)
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(pipe, args.out_model)
        json.dump(
            {"valid": valid_metrics, "params": {"C": C, "max_iter": max_iter, "seed": seed}},
            open(args.metrics, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

        # Log en MLflow: métricas simples + alias con prefijos valid_ y test_
        mlflow.log_params({"C": C, "max_iter": max_iter, "seed": seed})
        mlflow.log_metrics(
            {
                # nombres originales
                "accuracy": valid_metrics["accuracy"],
                "precision": valid_metrics["precision"],
                "recall": valid_metrics["recall"],
                "f1": valid_metrics["f1"],
                # alias "valid_*"
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1": valid_metrics["f1"],
                # alias "test_*" (sin conjunto test separado; igual a valid)
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
            f"[OK] valid: acc={valid_metrics['accuracy']:.4f} "
            f"prec={valid_metrics['precision']:.4f} "
            f"rec={valid_metrics['recall']:.4f} "
            f"f1={valid_metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
