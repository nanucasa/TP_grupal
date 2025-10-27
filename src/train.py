# src/train.py
# Entrenamiento baseline con scikit-learn + tracking en MLflow.
# Guarda modelo en models/model.joblib y métricas en metrics.json (para DVC).

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

def load_xy(path_csv: str, target: str = "churn"):
    df = pd.read_csv(path_csv)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out-model", required=True)   # e.g., models\model.joblib
    ap.add_argument("--metrics", required=True)     # e.g., metrics.json
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            C=args.C, max_iter=args.max_iter, solver="lbfgs", penalty="l2",
            random_state=args.seed, n_jobs=None, class_weight="balanced"))
    ])

    mlflow.set_experiment("telco_churn_baseline")
    mlflow.sklearn.autolog(log_input_examples=False, log_models=False)  # logeamos el modelo manualmente abajo

    with mlflow.start_run():
        pipe.fit(Xtr, ytr)

        prob = pipe.predict_proba(Xva)[:, 1]
        yhat = (prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0))
        }

        # guardar artefactos locales para DVC
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(pipe, args.out_model)

        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump({"valid": metrics, "params": {"C": args.C, "max_iter": args.max_iter, "seed": args.seed}},
                      f, ensure_ascii=False, indent=2)

        # log en MLflow
        mlflow.log_params({"C": args.C, "max_iter": args.max_iter, "seed": args.seed})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.metrics, artifact_path="reports")
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"[OK] valid: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
