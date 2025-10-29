
# Genera predicciones a partir de un CSV de features (puede incluir 'churn').
# Soporta umbral por valor directo (--threshold) o por archivo JSON (--threshold-file).

import argparse, json, os
import numpy as np, pandas as pd, joblib

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

def proba_from_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        # Normaliza a [0,1] para poder umbralizar
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    else:
        # Fallback: usa la predicción directa como 0/1 y la castea a float
        return model.predict(X).astype(float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threshold-file", default=None)
    ap.add_argument("--target", default="churn")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Si viene la columna target en el CSV, la removemos para predecir
    X = df.drop(columns=[args.target]) if args.target in df.columns else df.copy()

    model = joblib.load(args.model)
    prob = proba_from_model(model, X)

    thr = args.threshold if args.threshold is not None else load_threshold(args.threshold_file, 0.5)
    pred = (prob >= float(thr)).astype(int)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df = pd.DataFrame({"proba": prob, "pred": pred})
    out_df.to_csv(args.output, index=False)

    print(f"[OK] preds -> {args.output} | threshold={thr:.4f}")

if __name__ == "__main__":
    main()
