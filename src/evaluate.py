import argparse, json
import pandas as pd, numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.test)
    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"]).to_numpy(dtype=float)

    model = joblib.load(args.model)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        yhat = (prob >= 0.5).astype(int)
    else:
        prob = None
        yhat = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }
    if prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, prob))
        except Exception:
            pass

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"test": metrics}, f, ensure_ascii=False, indent=2)

    out = f"[OK] test: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    if "roc_auc" in metrics: out += f" roc_auc={metrics['roc_auc']:.4f}"
    print(out)

if __name__ == "__main__":
    main()
