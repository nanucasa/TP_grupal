
# Grid simple para XGBoost, maximiza F1 en valid, guarda mejores params y métricas.
import argparse, json, os
from itertools import product
import numpy as np, pandas as pd, yaml
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import mlflow

def load_xy(path, target="churn"):
    df = pd.read_csv(path)
    y = df[target].astype(int).to_numpy()
    X = df.drop(columns=[target])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--params", required=True)          # params.yaml
    ap.add_argument("--best-out", required=True)        # artifacts\best_params_xgb.json
    ap.add_argument("--metrics-out", required=True)     # metrics_tune_xgb.json
    args = ap.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)["tune_xgb"]
    grid = {
        "n_estimators": P["n_estimators"],
        "max_depth": P["max_depth"],
        "learning_rate": P["learning_rate"],
        "subsample": P["subsample"],
        "colsample_bytree": P["colsample_bytree"],
        "reg_lambda": P["reg_lambda"],
    }
    seed = int(P.get("random_state", 42))

    Xtr, ytr = load_xy(args.train)
    Xva, yva = load_xy(args.valid)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_tune_xgb")

    best = {"f1": -1.0, "params": None}
    for n_est, md, lr, ss, cs, rl in product(
        grid["n_estimators"], grid["max_depth"], grid["learning_rate"],
        grid["subsample"], grid["colsample_bytree"], grid["reg_lambda"]
    ):
        params = dict(
            n_estimators=int(n_est),
            max_depth=int(md),
            learning_rate=float(lr),
            subsample=float(ss),
            colsample_bytree=float(cs),
            reg_lambda=float(rl),
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
        )
        with mlflow.start_run():
            mlflow.log_params(params)
            clf = XGBClassifier(**params)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xva)[:, 1]
            yhat = (prob >= 0.5).astype(int)
            f1 = f1_score(yva, yhat, zero_division=0)
            mlflow.log_metrics({"valid_f1": float(f1)})

            if f1 > best["f1"]:
                best["f1"] = float(f1)
                best["params"] = {**params}

    os.makedirs(os.path.dirname(args.best_out), exist_ok=True)
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(best["params"], f, ensure_ascii=False, indent=2)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"best_valid_f1": best["f1"], "best_params": best["params"]}, f, ensure_ascii=False, indent=2)

    print(f"[OK][XGB] tune -> best {best['params']} | valid f1={best['f1']:.4f}")

if __name__ == "__main__":
    main()
