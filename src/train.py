# src/train.py
# Logistic Regression (scikit-learn) + MLflow (sin log_model) + params/best_params.
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, mlflow, mlflow.sklearn

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
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best-params", default="artifacts/best_params.json")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    # Defaults por si no hay params.yaml
    C_default, max_iter_default = 1.0, 1000
    try:
        import yaml
        if os.path.exists(args.params):
            with open(args.params, "r", encoding="utf-8") as f:
                P = (yaml.safe_load(f) or {})
            C_default = float(P.get("train", {}).get("C", C_default))
            max_iter_default = int(P.get("train", {}).get("max_iter", max_iter_default))
    except Exception:
        pass

    C = C_default if args.C is None else args.C
    max_iter = max_iter_default if args.max_iter is None else args.max_iter

    # Sobrescribir con best_params.json si existe
    if os.path.exists(args.best_params):
        try:
            with open(args.best_params, "r", encoding="utf-8") as f:
                best = json.load(f)
            C = float(best.get("C", C))
            max_iter = int(best.get("max_iter", max_iter))
        except Exception:
            pass

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", penalty="l2",
            class_weight="balanced", random_state=args.seed))
    ])

    # MLflow sin log_model (evitamos endpoint no soportado)
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

        # Guardar modelo local (DVC lo versiona)
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(pipe, args.out_model)

        # Guardar métricas locales (para DVC)
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump({"valid": metrics, "params": {"C": C, "max_iter": max_iter, "seed": args.seed}},
                      f, ensure_ascii=False, indent=2)

        # Log MLflow (sin log_model)
        mlflow.log_params({"C": C, "max_iter": max_iter, "seed": args.seed})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.metrics, artifact_path="reports")
        # subimos el .joblib como artifact genérico
        mlflow.log_artifact(args.out_model, artifact_path="model_files")

        print(f"[OK] valid: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
