
# sirve tanto si el server devuelve clases como probabilidades; aplica umbral si hace falta

import argparse
import json
import sys
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import requests


def load_threshold(th_json: str | None, th_value: float | None) -> float | None:
    if th_value is not None:
        return float(th_value)
    if th_json:
        try:
            with open(th_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # admite {"best_threshold": 0.45} o {"threshold": 0.45} o {"value": 0.45}
            for k in ("best_threshold", "threshold", "value"):
                if k in obj and isinstance(obj[k], (int, float)):
                    return float(obj[k])
        except Exception as e:
            print(f"[WARN] No pude leer umbral desde {th_json}: {e}", file=sys.stderr)
    return None


def to_pandas_split(df: pd.DataFrame) -> Dict[str, Any]:
    return {"columns": df.columns.tolist(), "data": df.values.tolist()}


def parse_predictions(resp: requests.Response) -> np.ndarray:
    # MLflow pyfunc suele devolver JSON con {"predictions": [...]}
    try:
        js = resp.json()
        if isinstance(js, dict) and "predictions" in js:
            preds = js["predictions"]
        else:
            # Puede devolver una lista directamente
            preds = js
    except Exception:
        # Si vino texto plano, intento parsear; si falla, devuelvo error legible
        txt = resp.text.strip()
        try:
            preds = json.loads(txt)
        except Exception:
            raise RuntimeError(f"Respuesta no parseable:\n{txt[:500]}")
    # Convertir a numpy
    arr = np.array(preds)
    return arr


def maybe_apply_threshold(arr: np.ndarray, threshold: float | None) -> pd.DataFrame:
    """
    Soporta:
      - vector 1D de clases (0/1) -> devuelve col 'pred'
      - vector 1D de probabilidades p(1) -> aplica umbral si se pasa, si no deja 'proba'
      - matriz 2D con proba para ambas clases (n,2) o (n,k) -> toma columna de la clase positiva si es binario
    """
    if arr.ndim == 1:
        # Puede ser clases o probas
        # Heurística: si valores dentro [0,1] y no son {0,1} puros -> probas
        vals = np.unique(arr[~np.isnan(arr)])
        if np.all((vals >= 0) & (vals <= 1)) and not np.array_equal(vals, np.array([0, 1])):
            proba = arr.astype(float)
            if threshold is not None:
                pred = (proba >= threshold).astype(int)
                return pd.DataFrame({"proba": proba, "pred": pred})
            else:
                return pd.DataFrame({"proba": proba})
        else:
            return pd.DataFrame({"pred": arr.astype(int)})
    elif arr.ndim == 2:
        # Si es binario y tiene 2 columnas, tomo la de la clase positiva (asumo col 1)
        if arr.shape[1] == 2:
            proba = arr[:, 1].astype(float)
            if threshold is not None:
                pred = (proba >= threshold).astype(int)
                return pd.DataFrame({"proba": proba, "pred": pred})
            else:
                return pd.DataFrame({"proba": proba})
        else:
            # Multiclase: devuelvo argmax
            pred = arr.argmax(axis=1)
            return pd.DataFrame({"pred": pred})
    else:
        raise ValueError(f"Dimensión de salida no soportada: {arr.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrada (e.g., data\\processed\\test.csv)")
    ap.add_argument("--url", default="http://127.0.0.1:9090/invocations", help="URL del endpoint /invocations")
    ap.add_argument("--limit", type=int, default=None, help="Cantidad de filas (head) a enviar. Si no, se envían todas.")
    ap.add_argument("--out", default=None, help="Ruta para guardar CSV con predicciones (opcional)")
    ap.add_argument("--threshold", type=float, default=None, help="Umbral explícito para convertir probas en clases")
    ap.add_argument("--threshold-json", default=None, help="Archivo JSON con el umbral (ej: artifacts\\best_threshold.json)")
    ap.add_argument("--drop-labels", nargs="*", default=["Churn", "Revenue"], help="Columnas a descartar si existen")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Descarta etiquetas si existen
    for col in args.drop_labels:
        if col in df.columns:
            df = df.drop(columns=[col])

    if args.limit is not None:
        df = df.head(args.limit)

    payload = to_pandas_split(df)

    # Determino umbral final (si corresponde)
    threshold = load_threshold(args.threshold_json, args.threshold)
    if threshold is not None:
        print(f"[INFO] Usando umbral = {threshold:.4f}")

    headers = {"Content-Type": "application/json; format=pandas-split"}
    try:
        resp = requests.post(args.url, headers=headers, data=json.dumps(payload), timeout=60)
    except Exception as e:
        print(f"[ERROR] No pude conectar al endpoint {args.url}: {e}", file=sys.stderr)
        sys.exit(2)

    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code}\n{resp.text[:1000]}", file=sys.stderr)
        sys.exit(3)

    try:
        arr = parse_predictions(resp)
        out_df = maybe_apply_threshold(arr, threshold)
    except Exception as e:
        print(f"[ERROR] Al parsear o postprocesar la respuesta: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"[OK] Recibidas {len(out_df)} predicciones")
    print(out_df.head(10).to_string(index=False))

    if args.out:
        out_path = args.out
        # Si solo vino 'proba' y hay threshold, genero 'pred' también
        if ("pred" not in out_df.columns) and (threshold is not None) and ("proba" in out_df.columns):
            out_df["pred"] = (out_df["proba"] >= threshold).astype(int)
        out_df.to_csv(out_path, index=False)
        print(f"[OK] Guardado CSV de salida en: {out_path}")


if __name__ == "__main__":
    main()
