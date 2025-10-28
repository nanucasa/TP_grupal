# Evalúa en test con umbral opcional (archivo o valor); guarda metrics_test*.json
import argparse, json, os
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

def load_threshold(path, default=0.5):
    t = default
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                J = json.load(f)
            for k in ("best_threshold", "threshold", "best", "value"):
                if k in J:
                    t = float(J[k])
                    break
        except Exception:
            pass
    return float(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threshold-file", default=None)
    ap.add_argument("--target", default="churn")
    args = ap.parse_args()

    df = pd.read_csv(args.test)
    y = df[args.target].astype(int).to_numpy()
    X = df.drop(columns=[args.target])

    clf = joblib.load(args.model)

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        s = np.asarray(clf.decision_function(X), dtype=float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)  # a [0,1]
        prob = s
    else:
        prob = clf.predict(X).astype(float)

    thr = args.threshold if args.threshold is not None else load_threshold(args.threshold_file, 0.5)
    yhat = (prob >= thr).astype(int)

    out = {
        "test": {
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y, yhat)),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
        }
    }
    try:
        out["test"]["roc_auc"] = float(roc_auc_score(y, prob))
    except Exception:
        pass
    try:
        out["test"]["pr_ap"] = float(average_precision_score(y, prob))
    except Exception:
        pass

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
