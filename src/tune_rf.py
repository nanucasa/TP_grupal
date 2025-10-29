# src/tune_rf.py
# Tuning de RandomForestClassifier y métricas en valid
import argparse, os, json, yaml
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

def load_xy(path, target="churn"):
    df = pd.read_csv(path)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--params", required=True)          # params.yaml
    ap.add_argument("--best-out", required=True)        # artifacts/best_params_rf.json
    ap.add_argument("--metrics-out", required=True)     # metrics_tune_rf.json
    args = ap.parse_args()

    P = yaml.safe_load(open(args.params, "r", encoding="utf-8"))
    T = P.get("tune_rf", {})
    n_grid   = T.get("n_estimators_grid", [100,200,300])
    d_grid   = T.get("max_depth_grid", [None,5,10,20])
    mss_grid = T.get("min_samples_split_grid", [2,5])
    cv = int(T.get("cv", 5)); scoring = str(T.get("scoring", "f1"))

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_tune_rf")

    grid = {
        "n_estimators": n_grid,
        "max_depth": d_grid,
        "min_samples_split": mss_grid,
    }
    base = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)

    with mlflow.start_run():
        gcv = GridSearchCV(base, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        gcv.fit(Xtr, ytr)
        best = gcv.best_estimator_
        prob = best.predict_proba(Xva)[:, 1]
        yhat = (prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }
        os.makedirs(os.path.dirname(args.best_out), exist_ok=True)
        json.dump({
            "n_estimators": int(best.n_estimators),
            "max_depth": None if best.max_depth is None else int(best.max_depth),
            "min_samples_split": int(best.min_samples_split),
        }, open(args.best_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump({"valid": metrics, "grid": grid}, open(args.metrics_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        mlflow.log_params({"cv": cv, "scoring": scoring, **{f"grid_{k}": v for k,v in grid.items()}})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.best_out, artifact_path="tuning_rf")
        mlflow.log_artifact(args.metrics_out, artifact_path="tuning_rf")
        print(f"[OK][RF] tune -> best {gcv.best_params_} | valid f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
