# src/evaluate.py
# Evalúa en TEST con el umbral óptimo (si existe). Guarda métricas en metrics_test.json
import argparse, os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import mlflow

def load_threshold(default=0.5, cli_thr=None):
    if cli_thr is not None:
        return float(cli_thr)
    path = os.path.join("artifacts", "best_threshold.json")
    if os.path.exists(path):
        try:
            return float(json.load(open(path, "r", encoding="utf-8"))["best_threshold"])
        except Exception:
            pass
    return float(default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)                  # data\processed\test.csv
    ap.add_argument("--model", required=True)                 # models\model.joblib
    ap.add_argument("--metrics-out", required=True)           # metrics_test.json
    ap.add_argument("--threshold", type=float, default=None)  # opcional
    args = ap.parse_args()

    # Datos
    df = pd.read_csv(args.test)
    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"]).to_numpy(dtype=float)

    # Modelo
    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("El modelo no expone predict_proba().")

    # Umbral
    thr = load_threshold(default=0.5, cli_thr=args.threshold)

    # Predicción y métricas
    prob = model.predict_proba(X)[:, 1]
    yhat = (prob >= thr).astype(int)

    metrics = {
        "test": {
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_ap": float(average_precision_score(y, prob))
        }
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Log en MLflow local (si está configurado)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_eval")
    with mlflow.start_run():
        mlflow.log_params({"threshold": thr})
        mlflow.log_metrics(metrics["test"])
        mlflow.log_artifact(args.metrics_out, artifact_path="evaluation")

    print(f"[OK] TEST | thr={thr:.4f} | f1={metrics['test']['f1']:.4f} "
          f"acc={metrics['test']['accuracy']:.4f} prec={metrics['test']['precision']:.4f} "
          f"rec={metrics['test']['recall']:.4f} roc_auc={metrics['test']['roc_auc']:.4f} pr_ap={metrics['test']['pr_ap']:.4f}")

if __name__ == "__main__":
    main()
