# Tuning para LogReg. Soporta holdout o CV (param tune.cv). Guarda mejores params y métricas.
import argparse, json, os
import numpy as np, pandas as pd, yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import mlflow

def load_xy(path, target="churn"):
    df = pd.read_csv(path)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y

def f1_from_model(clf, X, y):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
    else:
        s = clf.decision_function(X).astype(float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        p = s
    yhat = (p >= 0.5).astype(int)
    return f1_score(y, yhat, zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--params", required=True)       # params.yaml
    ap.add_argument("--best-out", required=True)     # artifacts\best_params.json
    ap.add_argument("--metrics-out", required=True)  # metrics_tune.json
    args = ap.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)
    T = P.get("tune", {})
    C_grid = T.get("C_grid", [0.01, 0.1, 1.0, 10.0])
    max_iter_grid = T.get("max_iter_grid", [1000])
    cv = int(T.get("cv", 1))  # <— default si no está en YAML
    seed = int(P.get("train", {}).get("seed", 42))

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "telco_churn_tune")
    mlflow.set_experiment(exp_name)

    best = {"score": -1.0, "params": None}

    for C in C_grid:
        for mi in max_iter_grid:
            params = dict(C=float(C), max_iter=int(mi), solver="liblinear", random_state=seed)
            with mlflow.start_run():
                mlflow.log_params(params)

                if cv > 1:
                    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
                    scores = []
                    for tr_idx, va_idx in skf.split(Xtr, ytr):
                        clf = LogisticRegression(**params)
                        clf.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
                        scores.append(f1_from_model(clf, Xtr.iloc[va_idx], ytr[va_idx]))
                    f1_cv = float(np.mean(scores))
                    mlflow.log_metric("valid_f1_cv", f1_cv)

                    # también reporto holdout (opcional)
                    clf = LogisticRegression(**params)
                    clf.fit(Xtr, ytr)
                    f1_holdout = float(f1_from_model(clf, Xva, yva))
                    mlflow.log_metric("valid_f1_holdout", f1_holdout)
                    score_use = f1_cv
                else:
                    clf = LogisticRegression(**params)
                    clf.fit(Xtr, ytr)
                    f1_holdout = float(f1_from_model(clf, Xva, yva))
                    mlflow.log_metric("valid_f1", f1_holdout)
                    score_use = f1_holdout

                if score_use > best["score"]:
                    best["score"] = score_use
                    best["params"] = {"C": params["C"], "max_iter": params["max_iter"]}

    os.makedirs(os.path.dirname(args.best_out), exist_ok=True)
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(best["params"], f, ensure_ascii=False, indent=2)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"best_valid_f1": best["score"], "best_params": best["params"], "cv": cv}, f, ensure_ascii=False, indent=2)

    print(f"[OK] tune -> best {best['params']} | valid f1={best['score']:.4f} | cv={cv}")

if __name__ == "__main__":
    main()
