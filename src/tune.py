# src/tune.py
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow, mlflow.sklearn

def load_xy(csv, target="churn"):
    df = pd.read_csv(csv)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--params", required=True)         # params.yaml leído por DVC, pero lo pasamos para dependencia
    ap.add_argument("--best-out", required=True)       # artifacts/best_params.json
    ap.add_argument("--metrics-out", required=True)    # metrics_tune.json
    args = ap.parse_args()

    # Cargar parámetros desde params.yaml (DVC registrará el cambio)
    import yaml
    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)
    C_grid = P["tune"]["C_grid"]
    cv = int(P["tune"]["cv"])
    scoring = str(P["tune"]["scoring"])
    max_iter = int(P["tune"]["max_iter"])

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, solver="lbfgs",
                                   penalty="l2", class_weight="balanced", random_state=42))
    ])
    grid = {"clf__C": C_grid}

    mlflow.set_experiment("telco_churn_tune")
    with mlflow.start_run():
        gcv = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
        gcv.fit(Xtr, ytr)

        best_C = float(gcv.best_params_["clf__C"])
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
        with open(args.best_out, "w", encoding="utf-8") as f:
            json.dump({"C": best_C, "max_iter": max_iter}, f, ensure_ascii=False, indent=2)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump({"valid": metrics, "grid": {"C": C_grid, "cv": cv, "scoring": scoring}},
                      f, ensure_ascii=False, indent=2)

        mlflow.log_params({"grid_C": C_grid, "cv": cv, "scoring": scoring, "max_iter": max_iter})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.best_out, artifact_path="tuning")
        mlflow.log_artifact(args.metrics_out, artifact_path="tuning")
        print(f"[OK] tune -> best C={best_C} | valid f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
