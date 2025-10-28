# src/train_rf.py
# Entrena RandomForest, loguea y registra en MLflow local con signature + input_example
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    ap.add_argument("--out-model", required=True)   # models\model_rf.joblib
    ap.add_argument("--metrics", required=True)     # metrics_rf.json
    ap.add_argument("--best-params", required=True) # artifacts\best_params_rf.json
    ap.add_argument("--params", required=True)      # params.yaml
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Defaults por YAML train_rf + override best_params_rf.json
    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)
    TR = P.get("train_rf", {})
    n_estimators = int(TR.get("n_estimators", 200))
    md = TR.get("max_depth", None)
    max_depth = None if md in (None, "null") else int(md)
    min_samples_split = int(TR.get("min_samples_split", 2))

    if os.path.exists(args.best_params):
        try:
            B = json.load(open(args.best_params, "r", encoding="utf-8"))
            n_estimators = int(B.get("n_estimators", n_estimators))
            md = B.get("max_depth", max_depth)
            max_depth = None if md in (None, "null") else int(md)
            min_samples_split = int(B.get("min_samples_split", min_samples_split))
        except Exception:
            pass

    Xtr_df, ytr, _ = load_xy(args.train)
    Xva_df, yva, _ = load_xy(args.valid)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )

    # MLflow local
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_baseline_rf")

    input_example = Xtr_df.head(5)
    with mlflow.start_run() as run:
        rf.fit(Xtr_df.to_numpy(dtype=float), ytr)
        prob = rf.predict_proba(Xva_df.to_numpy(dtype=float))[:, 1]
        yhat = (prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }

        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(rf, args.out_model)
        json.dump({"valid": metrics, "params": {
            "n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split,
            "seed": args.seed}}, open(args.metrics, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "seed": args.seed
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            input_example,
            rf.predict_proba(input_example.to_numpy(dtype=float))[:, 1]
        )
        mlflow.sklearn.log_model(
            rf,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        # Registrar en Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        reg_name = "TelcoChurn_RF"
        mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
        MlflowClient().transition_model_version_stage(
            name=reg_name, version=mv.version, stage="Staging", archive_existing_versions=False
        )

        print(f"[OK][RF] valid: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
