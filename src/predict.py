# src/predict.py
# Inferencia por lotes: recibe CSV (con o sin 'churn') y genera proba/pred.
import argparse, os, json
import pandas as pd, numpy as np, joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV de entrada; si existe 'churn', se descarta")
    ap.add_argument("--model", required=True, help="Ruta al modelo .joblib")
    ap.add_argument("--output", required=True, help="CSV de salida con proba y pred")
    ap.add_argument("--threshold", type=float, default=None, help="Umbral; si no se pasa, usa artifacts/best_threshold.json o 0.5")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

    X = df.to_numpy(dtype=float)
    model = joblib.load(args.model)

    # Umbral
    thr = args.threshold
    if thr is None:
        th_path = os.path.join("artifacts", "best_threshold.json")
        if os.path.exists(th_path):
            try:
                thr = float(json.load(open(th_path, "r", encoding="utf-8"))["best_threshold"])
            except Exception:
                thr = 0.5
        else:
            thr = 0.5

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= thr).astype(int)
    else:
        prob = None
        pred = model.predict(X)

    out = df.copy()
    if prob is not None:
        out["proba_churn"] = prob
    out["pred_churn"] = pred

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[OK] guardado: {args.output} ({len(out)} filas) | threshold={thr}")

if __name__ == "__main__":
    main()
