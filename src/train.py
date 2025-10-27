# src/train.py
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, mlflow, mlflow.sklearn

def load_xy(p, target="churn"):
    df = pd.read_csv(p); y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best-params", default="artifacts/best_params.json")
    ap.add_argument("--params", default="params.yaml")      # para que DVC pueda rastrear si queremos
    args = ap.parse_args()

    # por defecto leer de params.yaml
    import yaml
    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)
    C = P["train"]["C"] if args.C is None else args.C
    max_iter = P["train"]["max_iter"] if args.max_iter is None else args.max_iter

    # si existe best_params.json, usarlo
    if os.path.exists(args.best_params):
        with open(args.best_params, "r", encoding="utf-8") as f:
            best = json.load(f)
        C = float(best.get("C", C))
        max_iter = int(best.get("max_iter", max_iter))

    Xtr, ytr = load_xy(args.train); Xva, yva = load_xy(args.valid)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                                   penalty="l2", class_weight="balanced",
                                   random_state=args.seed))
    ])

    mlflow.set_experiment("telco_churn_baseline")
    mlflow.sklearn.autolog(log_input_examples=False, log_models=False)

    with mlflow.start_run():
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xva)[:, 1]
        yhat = (prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }

        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(pipe, args.out_model)
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump({"valid": metrics, "params": {"C": C, "max_iter": max_iter, "seed": args.seed}},
                      f, ensure_ascii=False, indent=2)
        mlflow.log_params({"C": C, "max_iter": max_iter, "seed": args.seed})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.metrics, artifact_path="reports")
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(f"[OK] valid: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
