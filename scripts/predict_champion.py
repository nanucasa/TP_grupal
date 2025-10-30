# scripts/predict_champion.py
import argparse, json, os, sys
import pandas as pd
import numpy as np
import joblib

try:
    import mlflow
    import mlflow.pyfunc
except Exception:
    mlflow = None  # toleramos ausencia si usamos modelo local

def load_threshold(th_file: str | None) -> float:
    if th_file and os.path.exists(th_file):
        with open(th_file, "r", encoding="utf-8") as f:
            J = json.load(f)
        for k in ["best_threshold", "threshold", "best_thr", "thr"]:
            if k in J:
                return float(J[k])
    return 0.5

def infer_uri_from_selection(sel: dict) -> tuple[str | None, str | None]:
    """
    Devuelve (tracking_uri, model_uri) si están en selection.json.
    Acepta varias formas:
      - {"mlflow_tracking_uri": "...", "mlflow_model_uri": "models:/Name/Version"}
      - {"mlflow_uri": "...", "model_uri": "models:/Name/Version"}
      - {"model_name": "TelcoChurn_XGB", "version": 9}
    """
    trk = sel.get("mlflow_tracking_uri") or sel.get("mlflow_uri")
    muri = sel.get("mlflow_model_uri") or sel.get("model_uri")

    if not muri and sel.get("model_name") and sel.get("version"):
        muri = f"models:/{sel['model_name']}/{sel['version']}"

    return trk, muri

def fallback_local_model() -> str | None:
    # orden de preferencia: XGB, RF, LogReg
    candidates = [
        os.path.join("models", "model_xgb.joblib"),
        os.path.join("models", "model_rf.joblib"),
        os.path.join("models", "model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="CSV con el set de test")
    ap.add_argument("--out", required=True, help="CSV de salida con predicciones")
    ap.add_argument("--selection", default=os.path.join("artifacts", "selection.json"),
                    help="selection.json con el campeón")
    ap.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    ap.add_argument("--threshold-file", default=None, help="JSON con best_threshold")
    ap.add_argument("--threshold", type=float, default=None, help="Umbral manual (sobrescribe threshold-file)")
    args = ap.parse_args()

    # Leer selección
    tracking_uri = args.mlflow_uri
    model_uri = None
    local_model_path = None

    if os.path.exists(args.selection):
        with open(args.selection, "r", encoding="utf-8") as f:
            sel = json.load(f)
        trk, muri = infer_uri_from_selection(sel)
        if trk:
            tracking_uri = trk
        if muri:
            model_uri = muri
        # opcionalmente permitimos ruta local directa
        if sel.get("local_model_path"):
            local_model_path = sel["local_model_path"]

    # Cargar modelo
    model = None
    used = ""
    if model_uri and mlflow is not None:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.pyfunc.load_model(model_uri)
            used = f"mlflow:{model_uri}"
        except Exception as e:
            print(f"[WARN] No se pudo cargar desde MLflow ({e}). Intento local...", file=sys.stderr)

    if model is None:
        if local_model_path and os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
            used = f"local:{local_model_path}"
        else:
            fpath = fallback_local_model()
            if not fpath:
                raise FileNotFoundError("No se encontró un modelo local en models/ ni se pudo cargar desde MLflow.")
            model = joblib.load(fpath)
            used = f"local:{fpath}"

    print(f"[INFO] Modelo usado -> {used}")

    # Umbral
    thr = args.threshold if args.threshold is not None else load_threshold(args.threshold_file)
    print(f"[INFO] Umbral = {thr:.4f}")

    # Datos
    df = pd.read_csv(args.test)
    # Intentamos preservar un id si existe
    id_col = None
    for cand in ["customerID", "customer_id", "id"]:
        if cand in df.columns:
            id_col = cand
            break

    # Quitar target si viene presente
    for tgt in ["Churn", "churn", "target", "Revenue"]:
        if tgt in df.columns:
            df = df.drop(columns=[tgt])

    X = df.copy()

    # Predicción
    proba = None
    yhat = None
    # si es pyfunc, predict puede devolver clase. Intentamos proba si existe
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, (list, tuple)):
                proba = np.asarray(proba)
            p1 = proba[:, 1]
            yhat = (p1 >= thr).astype(int)
        except Exception as e:
            print(f"[WARN] predict_proba falló ({e}), uso predict()", file=sys.stderr)
            yhat = model.predict(X)
            p1 = np.full(len(yhat), np.nan)
    else:
        try:
            preds = model.predict(X)
            yhat = preds if isinstance(preds, np.ndarray) else np.asarray(preds)
            p1 = np.full(len(yhat), np.nan)
        except Exception as e:
            raise RuntimeError(f"No se pudo predecir con el modelo: {e}")

    out = pd.DataFrame({
        **({id_col: df[id_col]} if id_col else {}),
        "proba": p1,
        "pred": yhat.astype(int)
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] Predicciones -> {args.out} | n={len(out)}")

if __name__ == "__main__":
    main()
