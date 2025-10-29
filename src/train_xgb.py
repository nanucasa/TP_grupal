
# Entrena XGBClassifier con mejores hiperparámetros, loguea y registra en MLflow con signature+input_example.
import argparse, os, json, yaml
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import mlflow, mlflow.xgboost
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
    ap.add_argument("--out-model", required=True)   # models\model_xgb.joblib
    ap.add_argument("--metrics", required=True)     # metrics_xgb.json
    ap.add_argument("--best-params", required=True) # artifacts\best_params_xgb.json
    ap.add_argument("--params", required=True)      # params.yaml
    args = ap.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)["train_xgb"]

    # Mezclar base + best
    best = {}
    if os.path.exists(args.best_params):
        try:
            best = json.load(open(args.best_params, "r", encoding="utf-8"))
        except Exception:
            best = {}
    cfg = {**base, **best}
    cfg.setdefault("tree_method", "hist")
    cfg.setdefault("n_jobs", -1)
    cfg.setdefault("random_state", int(base.get("random_state", 42)))
    cfg.setdefault("eval_metric", "logloss")

    Xtr_df, ytr, _ = load_xy(args.train)
    Xva_df, yva, _ = load_xy(args.valid)

    clf = XGBClassifier(**cfg)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_baseline_xgb")

    input_example = Xtr_df.head(5)
    with mlflow.start_run() as run:
        clf.fit(Xtr_df, ytr)
        prob = clf.predict_proba(Xva_df)[:, 1]
        yhat = (prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(yva, yhat)),
            "precision": float(precision_score(yva, yhat, zero_division=0)),
            "recall": float(recall_score(yva, yhat, zero_division=0)),
            "f1": float(f1_score(yva, yhat, zero_division=0)),
        }

        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        joblib.dump(clf, args.out_model)
        json.dump({"valid": metrics, "params": cfg}, open(args.metrics, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        mlflow.log_params(cfg)
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            input_example,
            clf.predict_proba(input_example)[:, 1]
        )
        mlflow.xgboost.log_model(
            clf,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        reg_name = "TelcoChurn_XGB"
        mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
        MlflowClient().transition_model_version_stage(
            name=reg_name, version=mv.version, stage="Staging", archive_existing_versions=False
        )

        print(f"[OK][XGB] valid: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
