# src/threshold.py
# Calcula el umbral óptimo (max F1 en valid), guarda métricas y curvas.
import argparse, os, json
import numpy as np, pandas as pd
import joblib
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import mlflow

def save_plot_pr(y_true, prob, path):
    try:
        import matplotlib.pyplot as plt
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure()
        plt.step(rec, prec, where="post")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve (AP={ap:.3f})")
        plt.savefig(path, bbox_inches="tight"); plt.close()
        return True
    except Exception:
        return False

def save_plot_roc(y_true, prob, path):
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, prob)
        rocauc = auc(fpr, tpr)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC curve (AUC={rocauc:.3f})")
        plt.savefig(path, bbox_inches="tight"); plt.close()
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-json", required=True)        # artifacts/best_threshold.json
    ap.add_argument("--metrics-out", required=True)     # metrics_threshold.json
    ap.add_argument("--roc-out", required=True)         # reports/roc_curve.png
    ap.add_argument("--pr-out", required=True)          # reports/pr_curve.png
    args = ap.parse_args()

    # Datos
    df = pd.read_csv(args.valid)
    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"]).to_numpy(dtype=float)

    # Modelo
    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("El modelo no expone predict_proba()")
    prob = model.predict_proba(X)[:, 1]

    # Curvas y umbral óptimo por F1
    prec, rec, thr = precision_recall_curve(y, prob)   # len(thr) = len(prec)-1
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.nanargmax(f1s))
    best_thr = float(thr[i])
    yhat = (prob >= best_thr).astype(int)

    # Métricas en valid con ese umbral
    metrics = {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "pr_ap": float(average_precision_score(y, prob)),
        "roc_auc": float(auc(*roc_curve(y, prob)[:2][::-1]))  # AUC(FPR,TPR)
    }

    # Guardados
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr}, f, ensure_ascii=False, indent=2)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"valid_threshold": metrics}, f, ensure_ascii=False, indent=2)

    _pr_ok = save_plot_pr(y, prob, args.pr_out)
    _roc_ok = save_plot_roc(y, prob, args.roc_out)

    # Log MLflow local (si está configurado)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("telco_churn_threshold")
    with mlflow.start_run():
        mlflow.log_params({"chosen_threshold": best_thr})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(args.out_json, artifact_path="threshold")
        mlflow.log_artifact(args.metrics_out, artifact_path="threshold")
        if _pr_ok:  mlflow.log_artifact(args.pr_out, artifact_path="threshold")
        if _roc_ok: mlflow.log_artifact(args.roc_out, artifact_path="threshold")

    print(f"[OK] best_threshold={best_thr:.4f} | valid f1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
